package com.echogesture.app

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.SdkSuppress
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.abs

@RunWith(AndroidJUnit4::class)
@SdkSuppress(minSdkVersion = 28)
class RecognitionParityTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val app: Context = instrumentation.targetContext
    private val testAssets = instrumentation.context.assets

    @Test
    fun originalSavedLandmarksMatchPythonIfAvailable() = compareKeypoints("saved")

    @Test
    fun landmarksRegeneratedFromReplayMatchPython() = compareKeypoints("replay")

    @Test
    fun replayThroughAndroidDecoderHolisticAndOnnx() {
        val reference = readReference("replay")
        assumeTrue("Real replay fixture is unavailable", "replay.mp4" in testAssets.list("").orEmpty())
        val clip = File.createTempFile("echo-parity-", ".mp4", app.cacheDir)
        try {
            testAssets.open("replay.mp4").use { source ->
                clip.outputStream().use { source.copyTo(it) }
            }
            val decoded = HolisticVideo(app).extract(clip)
            assertTrue("Expected usable hands in replay clip", decoded.hasHands)
            val result = LocalSignClassifier(app).use { it.inferKeypoints(decoded.raw) }
            val raw = reference.getJSONArray("raw")
            val differences = decoded.raw.indices.map { abs(decoded.raw[it] - raw.getDouble(it)) }
            val report = JSONObject()
                .put("frame_count", decoded.frameCount)
                .put("sampled_rows_with_hands", decoded.handRows)
                .put("raw_max_delta", differences.max())
                .put("raw_mean_delta", differences.average())
                .put("python_result", reference.getJSONObject("result"))
                .put("android_top5", JSONArray(result.topFive))
                .put("android_confidence", result.confidence)
            save("android-video-comparison.json", report)
        } finally {
            clip.delete()
        }
    }

    private fun compareKeypoints(name: String) {
        val reference = readReference(name)
        val raw = reference.getJSONArray("raw")
        val input = FloatArray(raw.length()) { raw.getDouble(it).toFloat() }
        val actual = LocalSignClassifier(app).use { it.inferKeypoints(input) }
        val features = reference.getJSONArray("engineered")
        assertEquals(features.length(), actual.features.size)
        val featureDelta = actual.features.indices.maxOf { abs(actual.features[it] - features.getDouble(it).toFloat()) }
        val expectedLogits = reference.getJSONArray("logits")
        assertEquals(expectedLogits.length(), actual.logits.size)
        val logitDelta = actual.logits.indices.map { row ->
            val referenceRow = expectedLogits.getJSONArray(row)
            actual.logits[row].indices.maxOf { index -> abs(actual.logits[row][index] - referenceRow.getDouble(index)) }
        }
        val combined = reference.getJSONArray("combined")
        val combinedDelta = actual.combined.indices.maxOf { abs(actual.combined[it] - combined.getDouble(it)) }
        val expectedResult = reference.getJSONObject("result")
        val top5 = expectedResult.getJSONArray("top5")
        val report = JSONObject()
            .put("feature_max_delta", featureDelta.toDouble())
            .put("per_model_logit_max_delta", JSONArray(logitDelta))
            .put("combined_logit_max_delta", combinedDelta)
            .put("python_result", expectedResult)
            .put("android_top5", JSONArray(actual.topFive))
            .put("android_confidence", actual.confidence)
        save("android-$name-keypoint-comparison.json", report)
        assertEquals(0f, featureDelta, 0f)
        assertTrue("Unexpected ONNX logits difference: $logitDelta", logitDelta.all { it <= 0.0002 })
        assertEquals(List(top5.length()) { top5.getString(it) }, actual.topFive)
        assertEquals(expectedResult.getDouble("confidence"), actual.confidence, 0.000051)
    }

    private fun readReference(name: String): JSONObject {
        val filename = "$name-python.json"
        assumeTrue("Missing Python fixture $filename", filename in testAssets.list("").orEmpty())
        return testAssets.open(filename).bufferedReader().use { JSONObject(it.readText()) }
    }

    private fun save(name: String, report: JSONObject) {
        File(app.filesDir, name).writeText(report.toString(2))
    }
}
