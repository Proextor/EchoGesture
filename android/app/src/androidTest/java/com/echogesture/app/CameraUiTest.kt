package com.echogesture.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.SystemClock
import android.widget.Button
import android.widget.TextView
import androidx.camera.view.PreviewView
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.SdkSuppress
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
@SdkSuppress(minSdkVersion = 28)
class CameraUiTest {
    @Test
    fun recordsOneSilentClipAndReturnsToReady() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val app = instrumentation.targetContext
        val expectRecognition = InstrumentationRegistry.getArguments().getString("cameraSignVideo") == "true"
        assumeTrue(app.packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY))
        instrumentation.uiAutomation.grantRuntimePermission(app.packageName, Manifest.permission.CAMERA)
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            await(scenario) { activity ->
                val preview = activity.findViewById<PreviewView>(R.id.camera_preview)
                preview.previewStreamState.value == PreviewView.StreamState.STREAMING &&
                    activity.findViewById<Button>(R.id.record_sign).isEnabled
            }
            scenario.onActivity { it.findViewById<Button>(R.id.record_sign).performClick() }
            SystemClock.sleep(1_500)
            scenario.onActivity { activity ->
                val button = activity.findViewById<Button>(R.id.record_sign)
                assertEquals("Finish recording", button.text.toString())
                button.performClick()
            }
            await(scenario) { activity ->
                activity.findViewById<TextView>(R.id.camera_status).text.toString() == "Capture complete" &&
                    activity.findViewById<Button>(R.id.record_sign).isEnabled
            }
            scenario.onActivity { activity ->
                val status = activity.findViewById<TextView>(R.id.recognition_status).text.toString()
                if (expectRecognition) {
                    val word = activity.findViewById<TextView>(R.id.recognized_word).text.toString()
                    assertEquals("", status)
                    assertTrue("Expected an ONNX prediction from the virtual camera", word.isNotBlank() && word != "—")
                    println("CameraX virtual camera prediction: $word")
                } else {
                    assertEquals("No usable hands found. Record the whole sign and try again.", status)
                }
                assertTrue(activity.cacheDir.listFiles().orEmpty().none { it.name.startsWith("echo-sign-") })
            }
        }
    }

    private fun await(scenario: ActivityScenario<MainActivity>, condition: (MainActivity) -> Boolean) {
        val deadline = SystemClock.elapsedRealtime() + 45_000
        while (SystemClock.elapsedRealtime() < deadline) {
            var ready = false
            scenario.onActivity { ready = condition(it) }
            if (ready) return
            SystemClock.sleep(100)
        }
        throw AssertionError("Timed out waiting for the camera UI")
    }
}
