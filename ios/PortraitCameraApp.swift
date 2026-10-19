import SwiftUI
import AVFoundation
import Vision
import CoreML
import CoreImage
import Metal
import MetalKit
import UIKit

@main
struct PortraitCameraApp: App {
    var body: some Scene { WindowGroup { CameraScreen() } }
}
struct CameraScreen: View {
    @StateObject private var camera = CameraEngine()
    var body: some View {
        VStack {
            if let image = camera.image {
                Image(uiImage: image).resizable().scaledToFit()
            } else { Text("Camera preview") }
            Text(camera.status).font(.caption).padding()
        }
        .onAppear { camera.start() }
        .onDisappear { camera.stop() }
    }
}

// Use Swift 5 language mode for this prototype. Session and request work is serialized.
final class CameraEngine: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var image: UIImage?
    @Published var status = "Waiting for camera permission"
    private let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "portrait.camera")
    private let context = CIContext()
    private var configured = false
    private var renderer: MetalBlend?
    private var request: VNRequest?
    private var frames = 0
    private var dropped = 0
    private var milliseconds: [Double] = []

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] allowed in
            guard let self = self else { return }
            guard allowed else { self.publish("Camera access denied. Enable it in Settings."); return }
            self.queue.async {
                do {
                    if !self.configured { try self.configure() }
                    if !self.session.isRunning { self.session.startRunning() }
                } catch { self.publish("Setup error: \(error.localizedDescription)") }
            }
        }
    }
    func stop() { queue.async { if self.session.isRunning { self.session.stopRunning() } } }
    private func publish(_ text: String) { DispatchQueue.main.async { self.status = text } }
    private func configure() throws {
        renderer = try MetalBlend()
        // Optional custom model: export AlphaNet, add AlphaNet.mlpackage to the target.
        // Synthetic-trained AlphaNet is only a conversion smoke test, not a person model.
        if let url = Bundle.main.url(forResource: "AlphaNet", withExtension: "mlmodelc") {
            let model = try MLModel(contentsOf: url)
            let custom = VNCoreMLRequest(model: try VNCoreMLModel(for: model))
            custom.imageCropAndScaleOption = .scaleFill
            request = custom
        } else {
            let person = VNGeneratePersonSegmentationRequest()
            person.qualityLevel = .balanced
            person.outputPixelFormat = kCVPixelFormatType_OneComponent8
            request = person
        }
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) else {
            throw NSError(domain: "Portrait", code: 1, userInfo: [NSLocalizedDescriptionKey:"Front camera unavailable"])
        }
        let input = try AVCaptureDeviceInput(device: camera)
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = .vga640x480
        guard session.canAddInput(input), session.canAddOutput(output) else {
            throw NSError(domain: "Portrait", code: 2, userInfo: [NSLocalizedDescriptionKey:"Camera configuration unsupported"])
        }
        session.addInput(input); session.addOutput(output)
        if let connection = output.connection(with: .video) {
            if connection.isVideoOrientationSupported { connection.videoOrientation = .portrait }
            if connection.isVideoMirroringSupported {
                connection.automaticallyAdjustsVideoMirroring = false
                connection.isVideoMirrored = true
            }
        }
        configured = true
    }
    func captureOutput(_ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        dropped += 1
    }
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer), let request = request, let renderer = renderer else { return }
        let start = CACurrentMediaTime()
        do {
            try VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .up).perform([request])
            guard let mask = request.results?.first as? VNPixelBufferObservation else {
                publish("No mask image returned"); return
            }
            let texture = try renderer.render(frame: buffer, mask: mask.pixelBuffer)
            guard let ci = CIImage(mtlTexture: texture, options: [.colorSpace: CGColorSpaceCreateDeviceRGB()]),
                  let cg = context.createCGImage(ci, from: ci.extent) else { return }
            // Metal texture coordinates are top-left; CI image coordinates are bottom-left.
            let display = UIImage(cgImage: cg, scale: 1, orientation: .downMirrored)
            frames += 1
            milliseconds.append((CACurrentMediaTime()-start)*1000)
            if milliseconds.count > 300 { milliseconds.removeFirst() }
            let sorted = milliseconds.sorted()
            let p50 = sorted[sorted.count/2]
            let p95 = sorted[min(sorted.count-1,Int(Double(sorted.count-1)*0.95))]
            let text = String(format:"Processing p50 %.1f / p95 %.1f ms | frames %d | capture drops %d | thermal %@",p50,p95,frames,dropped,String(describing: ProcessInfo.processInfo.thermalState))
            DispatchQueue.main.async { self.image = display; self.status = text }
        } catch { publish("Frame error: \(error.localizedDescription)") }
    }
}

final class MetalBlend {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private var cache: CVMetalTextureCache?
    init() throws {
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue(),
              let function = d.makeDefaultLibrary()?.makeFunction(name:"portraitBlend") else {
            throw NSError(domain:"Portrait",code:3,userInfo:[NSLocalizedDescriptionKey:"Metal shader unavailable; include Blend.metal in target"])
        }
        device=d; queue=q; pipeline=try d.makeComputePipelineState(function:function)
        let result=CVMetalTextureCacheCreate(nil,nil,d,nil,&cache)
        if result != kCVReturnSuccess { throw NSError(domain:"Portrait",code:4) }
    }
    private func texture(_ buffer: CVPixelBuffer, format: MTLPixelFormat) throws -> CVMetalTexture {
        var result: CVMetalTexture?
        let code=CVMetalTextureCacheCreateTextureFromImage(nil,cache!,buffer,nil,format,CVPixelBufferGetWidth(buffer),CVPixelBufferGetHeight(buffer),0,&result)
        guard code == kCVReturnSuccess, let result=result else { throw NSError(domain:"Portrait",code:5,userInfo:[NSLocalizedDescriptionKey:"Unsupported pixel buffer for Metal"]) }
        return result
    }
    func render(frame: CVPixelBuffer, mask: CVPixelBuffer) throws -> MTLTexture {
        let f=try texture(frame,format:.bgra8Unorm), m=try texture(mask,format:.r8Unorm)
        let descriptor=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.rgba8Unorm,width:CVPixelBufferGetWidth(frame),height:CVPixelBufferGetHeight(frame),mipmapped:false)
        descriptor.usage=[.shaderRead,.shaderWrite]; descriptor.storageMode = .shared
        guard let out=device.makeTexture(descriptor:descriptor), let command=queue.makeCommandBuffer(), let encoder=command.makeComputeCommandEncoder() else { throw NSError(domain:"Portrait",code:6) }
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(CVMetalTextureGetTexture(f),index:0); encoder.setTexture(CVMetalTextureGetTexture(m),index:1); encoder.setTexture(out,index:2)
        let w=pipeline.threadExecutionWidth, h=max(1,pipeline.maxTotalThreadsPerThreadgroup/w)
        encoder.dispatchThreads(MTLSize(width:out.width,height:out.height,depth:1),threadsPerThreadgroup:MTLSize(width:w,height:h,depth:1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        if let error=command.error { throw error }
        // Retain Core Video wrappers until GPU work has completed.
        withExtendedLifetime((f,m)) {}
        return out
    }
}
