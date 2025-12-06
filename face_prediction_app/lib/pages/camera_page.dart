import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;

class CameraPage extends StatefulWidget {
  const CameraPage({super.key});

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  CameraController? controller;
  List<CameraDescription>? cameras;
  bool isReady = false;

  @override
  void initState() {
    super.initState();
    initCamera();
  }

  Future<void> initCamera() async {
    cameras = await availableCameras();

    controller = CameraController(
      cameras![0], // kamera belakang
      ResolutionPreset.medium,
      enableAudio: false,
    );

    await controller!.initialize();

    if (!mounted) return;

    setState(() {
      isReady = true;
    });
  }

  @override
  void dispose() {
    controller?.dispose();
    super.dispose();
  }

  // ======================================
  //  UPLOAD IMAGE KE API PYTHON / API LAIN
  // ======================================
  Future<String> uploadImage(File imageFile) async {
    var request = http.MultipartRequest(
      "POST",
      Uri.parse("http://10.0.2.2:5000/upload"), 
      // "10.0.2.2" = localhost untuk Android Emulator
    );

    request.files.add(
      await http.MultipartFile.fromPath('file', imageFile.path),
    );

    var response = await request.send();
    String result = await response.stream.bytesToString();
    return result;
  }

  @override
  Widget build(BuildContext context) {
    if (!isReady) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(title: const Text("Camera Preview")),
      body: CameraPreview(controller!),
      floatingActionButton: FloatingActionButton(
        onPressed: () async {
          final picture = await controller!.takePicture();

          // Pindah ke halaman preview sambil upload
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (_) => PreviewPage(imagePath: picture.path),
            ),
          );
        },
        child: const Icon(Icons.camera_alt),
      ),
    );
  }
}

// ======================================
// HALAMAN PREVIEW FOTO + UPLOAD IMAGE
// ======================================
class PreviewPage extends StatefulWidget {
  final String imagePath;

  const PreviewPage({super.key, required this.imagePath});

  @override
  State<PreviewPage> createState() => _PreviewPageState();
}

class _PreviewPageState extends State<PreviewPage> {
  bool isUploading = false;
  String uploadResult = "";

  Future<void> upload() async {
    setState(() => isUploading = true);

    var request = http.MultipartRequest(
      "POST",
      Uri.parse("http://10.0.2.2:5000/upload"),
    );

    request.files.add(
      await http.MultipartFile.fromPath('file', widget.imagePath),
    );

    var response = await request.send();
    uploadResult = await response.stream.bytesToString();

    setState(() => isUploading = false);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Preview Foto")),
      body: Column(
        children: [
          Expanded(
            child: Image.file(File(widget.imagePath)),
          ),

          const SizedBox(height: 10),

          isUploading
              ? const CircularProgressIndicator()
              : ElevatedButton.icon(
                  onPressed: upload,
                  icon: const Icon(Icons.cloud_upload),
                  label: const Text("Upload ke API"),
                ),

          if (uploadResult.isNotEmpty)
            Padding(
              padding: const EdgeInsets.all(12.0),
              child: Text(
                "Response API: $uploadResult",
                style: const TextStyle(fontSize: 14),
              ),
            ),
        ],
      ),
    );
  }
}
