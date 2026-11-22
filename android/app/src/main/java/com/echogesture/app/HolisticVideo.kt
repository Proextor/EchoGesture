package com.echogesture.app

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import androidx.annotation.RequiresApi
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarker
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarkerResult
import java.io.File

internal data class VideoLandmarks(val raw: FloatArray, val frameCount: Int, val handRows: Int) {
    val hasHands get() = handRows > 0
}

@RequiresApi(28)
internal class HolisticVideo(private val context: Context) {
    fun extract(clip: File): VideoLandmarks {
        val decoder = MediaMetadataRetriever()
        try {
            decoder.setDataSource(clip.absolutePath)
            val count = decoder.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)
                ?.toIntOrNull()?.takeIf { it > 0 } ?: error("Video frame count is unavailable")
            val indices = LegacyFeatures.sampleIndices(count)
            val raw = FloatArray(LegacyFeatures.FRAMES * LegacyFeatures.RAW_SIZE)
            val options = HolisticLandmarker.HolisticLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath("holistic_landmarker.task").build())
                .setRunningMode(RunningMode.VIDEO)
                .setMinPoseDetectionConfidence(0.5f)
                .setMinPosePresenceConfidence(0.5f)
                .setMinHandLandmarksConfidence(0.5f)
                .setOutputPoseSegmentationMasks(false)
                .setOutputFaceBlendshapes(false)
                .build()
            var row = 0
            var handRows = 0
            var decodedFrames = 0
            HolisticLandmarker.createFromOptions(context, options).use { holistic ->
                // Track each decoded frame in order, including unsampled frames.
                for (first in 0 until count step 8) {
                    val frames = decoder.getFramesAtIndex(first, minOf(8, count - first))
                    try {
                        for ((offset, frame) in frames.withIndex()) {
                            val frameIndex = first + offset
                            decodedFrames++
                            val result = BitmapImageBuilder(frame).build().use { image ->
                                holistic.detectForVideo(image, (frameIndex + 1L) * 33_333 / 1000)
                            }
                            while (row < indices.size && indices[row] < frameIndex) row++
                            if (row >= indices.size || indices[row] != frameIndex) continue
                            val points = landmarkRow(result)
                            val hasHands = (0 until 42).any { LegacyFeatures.present(points, it * 3) }
                            while (row < indices.size && indices[row] == frameIndex) {
                                points.copyInto(raw, row * LegacyFeatures.RAW_SIZE)
                                if (hasHands) handRows++
                                row++
                            }
                        }
                    } finally {
                        frames.forEach(Bitmap::recycle)
                    }
                }
            }
            check(decodedFrames > 0) { "No video frames decoded" }
            return VideoLandmarks(raw, count, handRows)
        } finally {
            decoder.release()
        }
    }

    private fun landmarkRow(result: HolisticLandmarkerResult): FloatArray {
        val row = FloatArray(LegacyFeatures.RAW_SIZE)
        result.leftHandLandmarks().forEachIndexed { i, point -> row.write(i, point) }
        result.rightHandLandmarks().forEachIndexed { i, point -> row.write(21 + i, point) }
        val pose = result.poseLandmarks()
        if (pose.isNotEmpty()) {
            LegacyFeatures.poseIds.forEachIndexed { i, id -> row.write(42 + i, pose[id]) }
            LegacyFeatures.normalize(row)
        }
        return row
    }

    private fun FloatArray.write(index: Int, point: NormalizedLandmark) {
        val offset = index * 3
        this[offset] = point.x()
        this[offset + 1] = point.y()
        this[offset + 2] = point.z()
    }
}
