# iPhone prototype: SOURCE ONLY, not yet compiled or device-tested

Business purpose: suppress distracting backgrounds in camera previews while
measuring processing latency. This app replaces the background with a solid
color. It does not reproduce RVM or depth-based bokeh.

## Run on a Mac and your iPhone
1. In Xcode create an iOS App named PortraitCamera, SwiftUI interface, Swift language.
2. Set iOS deployment target to 15 or later and Swift language mode to Swift 5.
3. Remove the generated App and ContentView Swift files. Add `PortraitCameraApp.swift`
   and `Blend.metal` to the app target, with target membership enabled.
4. Add `Privacy - Camera Usage Description` (`NSCameraUsageDescription`) to the
   target Info settings: "Camera frames are processed locally to demonstrate portrait effects."
5. Under Signing & Capabilities select your development team and a unique bundle ID.
6. Set supported iPhone orientations to portrait. Connect your iPhone and select it
   as the run destination. Run on the physical phone, allow camera access.

The default path uses Apple's Vision person segmentation, then our Metal compute
shader. It needs no model download. The app does not save or upload camera frames.
Camera work runs on one serial queue; late capture frames are discarded. GPU work
is deliberately synchronous for initial correctness, so it is not optimized.
The preview performs a CPU image conversion and main-thread update every frame.
This can introduce extra latency; the displayed timing excludes final screen
presentation and is not camera-to-display latency. Capture drop count does not
count delayed/coalesced UI updates. It displays thermal state, not energy usage.

## Optional Core ML path
On your Mac, install `coremltools` in a compatible Python environment alongside
PyTorch. From the repository root:

```
python -m experiments.export_coreml --checkpoint results/training_demo/best.pt --output AlphaNet.mlpackage
```

Add AlphaNet.mlpackage to the Xcode target. The app uses it when AlphaNet.mlmodelc
is in the bundle; otherwise it uses Vision. The demo checkpoint learned synthetic
circles, NOT people. Use it only to check conversion and integration, then remove
it for the Vision person demo. A real-data-trained model needs its own held-out
quality assessment. Core ML conversion and this Swift/Metal source have NOT been
executed here. Do not claim Apple Neural Engine use just because Core ML is used.

## Device validation to record
- Portrait preview orientation and mask alignment (including front-camera mirroring).
- Hair, fingers, rapid motion, empty scene and return to scene.
- Permission denial, background/foreground, and reopening the view.
- At least 5 minutes of p50/p95 processing times, capture drops and thermal states.
- Compare rendered pixels with a CPU blend before claiming Metal parity/speedup.

The `.downMirrored` display orientation must be checked on hardware; adjust only
if the captured-buffer-to-display mapping differs. This source does not establish
ARKit, SLAM or production camera engineering experience.

References:
https://developer.apple.com/documentation/vision/vngeneratepersonsegmentationrequest
https://developer.apple.com/documentation/vision/applying-matte-effects-to-people-in-images-and-video
https://apple.github.io/coremltools/docs-guides/source/image-inputs.html
