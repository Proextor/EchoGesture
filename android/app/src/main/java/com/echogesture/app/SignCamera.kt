package com.echogesture.app

import android.annotation.SuppressLint
import android.view.Surface
import androidx.activity.ComponentActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FallbackStrategy
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import java.io.File
import java.io.IOException

sealed interface CaptureOutcome {
    data class Ready(val clip: File) : CaptureOutcome
    data class Message(val text: String) : CaptureOutcome
}

class SignCamera(
    private val activity: ComponentActivity,
    private val view: PreviewView,
    private val onCameraState: (ready: Boolean, message: String) -> Unit,
    private val onCapture: (CaptureOutcome) -> Unit,
) {
    private val executor = ContextCompat.getMainExecutor(activity)
    private var recorder: VideoCapture<Recorder>? = null
    private var recording: Recording? = null
    private var binding = false
    private var discard = false

    fun bind() {
        if (binding || recorder != null) return
        binding = true
        onCameraState(false, "Starting camera…")
        val provider = ProcessCameraProvider.getInstance(activity)
        provider.addListener({
            binding = false
            if (activity.isDestroyed) return@addListener
            try {
                val cameraProvider = provider.get()
                val selector = if (cameraProvider.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA)) {
                    CameraSelector.DEFAULT_FRONT_CAMERA
                } else {
                    CameraSelector.DEFAULT_BACK_CAMERA
                }
                val preview = Preview.Builder().build().apply {
                    setSurfaceProvider(view.surfaceProvider)
                }
                val capture = VideoCapture.withOutput(
                    Recorder.Builder().setQualitySelector(QualitySelector.from(
                        Quality.SD, FallbackStrategy.higherQualityOrLowerThan(Quality.SD)
                    )).build()
                )
                cameraProvider.bindToLifecycle(activity, selector, preview, capture)
                recorder = capture
                onCameraState(true, "Camera ready")
            } catch (_: Exception) {
                onCameraState(false, "Camera unavailable. Try again.")
            }
        }, executor)
    }

    @SuppressLint("MissingPermission") // The activity checks permission before calling start().
    fun start(): Boolean {
        val capture = recorder ?: return false
        if (recording != null || !activity.lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)) return false
        val clip = try {
            File.createTempFile("echo-sign-", ".mp4", activity.cacheDir)
        } catch (_: IOException) {
            onCapture(CaptureOutcome.Message("Could not create a recording."))
            return false
        }
        discard = false
        capture.targetRotation = view.display?.rotation ?: Surface.ROTATION_0
        return try {
            recording = capture.output.prepareRecording(activity, FileOutputOptions.Builder(clip).build())
                .start(executor) { event ->
                    if (event is VideoRecordEvent.Finalize) finalize(clip, event)
                }
            true
        } catch (_: IllegalStateException) {
            clip.delete()
            onCapture(CaptureOutcome.Message("Recording could not start. Try again."))
            false
        } catch (_: SecurityException) {
            clip.delete()
            onCapture(CaptureOutcome.Message("Camera permission is required."))
            false
        }
    }

    fun finish() {
        recording?.stop()
    }

    fun discardInProgress() {
        discard = true
        finish()
    }

    private fun finalize(clip: File, event: VideoRecordEvent.Finalize) {
        recording = null
        val stats = event.recordingStats
        val validity = captureValidity(minOf(stats.numBytesRecorded, clip.length()), stats.recordedDurationNanos)
        val message = when {
            discard || activity.isDestroyed -> "Recording cancelled."
            event.error == VideoRecordEvent.Finalize.ERROR_NO_VALID_DATA -> "Empty capture. Record one sign."
            event.hasError() -> "Recording failed. Try again."
            validity == CaptureValidity.EMPTY -> "Empty capture. Record one sign."
            validity == CaptureValidity.TOO_SHORT -> "Capture too short. Record the whole sign."
            else -> null
        }
        if (message == null) {
            onCapture(CaptureOutcome.Ready(clip))
        } else {
            clip.delete()
            if (!activity.isDestroyed) onCapture(CaptureOutcome.Message(message))
        }
    }
}
