// CameraController.swift
// Phase 3a — owns an AVCaptureSession + AVCapturePhotoOutput, wired so
// CameraView can show a live preview and call `capturePhoto(...)` to
// get a real `UIImage` back.
//
// LIFECYCLE:
//   • init() does NOT touch hardware — safe to construct on main.
//   • prepare() requests camera authorization and configures the session.
//     Idempotent: calling twice is a no-op after the first success.
//   • startSession() / stopSession() flip the running flag. Hardware
//     stays held only between these — caller's job to stop on disappear.
//   • capturePhoto(...) hands back a downscaled UIImage (max 1280 px on
//     long edge) on the main queue. ~3 MB instead of ~36 MB raw, which
//     keeps total app footprint comfortable while Gemma + Kokoro are
//     also resident.
//
// SIMULATOR NOTE: AVCaptureDevice.default(.builtInWideAngleCamera, ...)
// returns nil on iOS Simulator — there's no camera hardware. In that
// case `prepare()` sets `error = .noCameraAvailable` and the UI falls
// back to a friendly message.

import AVFoundation
import UIKit

@MainActor
final class CameraController: NSObject, ObservableObject {

    // MARK: - Published state

    enum AuthorizationState: Equatable {
        case notDetermined
        case authorized
        case denied
        case restricted
    }

    enum CameraError: LocalizedError, Equatable {
        case noCameraAvailable
        case configurationFailed
        case captureFailed(String)

        var errorDescription: String? {
            switch self {
            case .noCameraAvailable:
                return "No camera available on this device."
            case .configurationFailed:
                return "Couldn't configure the camera."
            case .captureFailed(let s):
                return "Capture failed: \(s)"
            }
        }
    }

    @Published private(set) var authorization: AuthorizationState = .notDetermined
    @Published private(set) var isReady: Bool = false
    @Published private(set) var isRunning: Bool = false
    @Published private(set) var error: CameraError?

    // MARK: - Internals

    /// Exposed only so CameraPreviewView can hand it to AVCaptureVideoPreviewLayer.
    let session = AVCaptureSession()

    private let photoOutput = AVCapturePhotoOutput()
    private let sessionQueue = DispatchQueue(label: "com.lijuncheng16.HikeCompanion.camera")

    /// Stored at capture time, invoked from the photo delegate callback
    /// after the JPEG has been decoded + downscaled.
    private var pendingCompletion: ((UIImage?) -> Void)?

    private var didConfigure: Bool = false

    // MARK: - Public API

    /// Request camera authorization (if needed) and configure the session.
    /// Safe to call multiple times.
    func prepare() {
        let current = AVCaptureDevice.authorizationStatus(for: .video)
        switch current {
        case .authorized:
            authorization = .authorized
            configureIfNeeded()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                Task { @MainActor in
                    guard let self else { return }
                    self.authorization = granted ? .authorized : .denied
                    if granted { self.configureIfNeeded() }
                }
            }
        case .denied:
            authorization = .denied
        case .restricted:
            authorization = .restricted
        @unknown default:
            authorization = .denied
        }
    }

    /// Spin up the session. No-op if not yet configured.
    func startSession() {
        guard didConfigure else { return }
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if !self.session.isRunning {
                self.session.startRunning()
                Task { @MainActor in self.isRunning = true }
            }
        }
    }

    /// Stop the session — releases the camera hardware so the green LED
    /// goes off and other apps can use it.
    func stopSession() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if self.session.isRunning {
                self.session.stopRunning()
                Task { @MainActor in self.isRunning = false }
            }
        }
    }

    /// Capture a single photo and call `completion` on the main queue
    /// with a downscaled UIImage (or nil on failure).
    func capturePhoto(completion: @escaping (UIImage?) -> Void) {
        guard isReady, isRunning else {
            completion(nil)
            return
        }
        // Make sure only one capture is in flight at a time.
        guard pendingCompletion == nil else {
            completion(nil)
            return
        }
        pendingCompletion = completion

        sessionQueue.async { [weak self] in
            guard let self else { return }
            let settings = AVCapturePhotoSettings()
            // Use auto flash only if the device supports it (iPad Air has no flash).
            if self.photoOutput.supportedFlashModes.contains(.auto) {
                settings.flashMode = .auto
            }
            self.photoOutput.capturePhoto(with: settings, delegate: self)
        }
    }

    // MARK: - Configuration

    private func configureIfNeeded() {
        guard !didConfigure else { return }
        sessionQueue.async { [weak self] in
            guard let self else { return }

            self.session.beginConfiguration()
            self.session.sessionPreset = .photo

            // Wide-angle back camera. Ultra-wide / telephoto are nice
            // but we want the most universally available device.
            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                       for: .video,
                                                       position: .back) else {
                Task { @MainActor in self.error = .noCameraAvailable }
                self.session.commitConfiguration()
                return
            }

            do {
                let input = try AVCaptureDeviceInput(device: device)
                if self.session.canAddInput(input) {
                    self.session.addInput(input)
                } else {
                    throw CameraError.configurationFailed
                }
            } catch {
                Task { @MainActor in self.error = .configurationFailed }
                self.session.commitConfiguration()
                return
            }

            if self.session.canAddOutput(self.photoOutput) {
                self.session.addOutput(self.photoOutput)
            } else {
                Task { @MainActor in self.error = .configurationFailed }
                self.session.commitConfiguration()
                return
            }

            self.session.commitConfiguration()
            self.didConfigure = true
            Task { @MainActor in self.isReady = true }
        }
    }
}

// MARK: - AVCapturePhotoCaptureDelegate

extension CameraController: AVCapturePhotoCaptureDelegate {
    nonisolated func photoOutput(_ output: AVCapturePhotoOutput,
                                 didFinishProcessingPhoto photo: AVCapturePhoto,
                                 error: Error?) {
        // Decode + downscale on this background callback queue, then
        // hop to main to deliver the result.
        let result: UIImage? = {
            if let error {
                _ = error
                return nil
            }
            guard let data = photo.fileDataRepresentation(),
                  let img = UIImage(data: data) else {
                return nil
            }
            return Self.downscale(img, maxLongEdge: 1280)
        }()

        Task { @MainActor in
            let cb = self.pendingCompletion
            self.pendingCompletion = nil
            MemoryStats.log("camera.capture done")
            cb?(result)
        }
    }

    /// Resize an image so that the long edge fits in `maxLongEdge` pts,
    /// preserving aspect ratio + orientation. Photos from a 12 MP back
    /// camera are ~36 MB raw; this brings them to ~3 MB while still
    /// looking sharp in the photo-context thumbnail.
    nonisolated private static func downscale(_ image: UIImage, maxLongEdge: CGFloat) -> UIImage {
        let w = image.size.width, h = image.size.height
        let long = max(w, h)
        guard long > maxLongEdge else { return image }

        let scale = maxLongEdge / long
        let newSize = CGSize(width: w * scale, height: h * scale)

        // Use a non-opaque renderer so transparent images survive
        // (camera output is opaque, but it's cheap insurance).
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1.0
        let renderer = UIGraphicsImageRenderer(size: newSize, format: format)
        return renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: newSize))
        }
    }
}
