package com.echogesture.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Color
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.text.InputType
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.widget.doAfterTextChanged
import org.json.JSONArray
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    private val classifier: SignClassifier by lazy { LocalSignClassifier(applicationContext) }
    private val worker = Executors.newSingleThreadExecutor()
    private val preferences by lazy { getSharedPreferences("sentences", MODE_PRIVATE) }
    private lateinit var camera: SignCamera
    private lateinit var speech: OfflineSpeech
    private lateinit var cameraButton: Button
    private lateinit var recordButton: Button
    private lateinit var cameraStatus: TextView
    private lateinit var modelStatus: TextView
    private lateinit var wordView: TextView
    private lateinit var addWord: Button
    private lateinit var sentence: EditText
    private lateinit var speechStatus: TextView
    private lateinit var historyView: LinearLayout
    private var pendingWord: String? = null
    private var cameraReady = false
    private var capturing = false
    private var busy = false
    private var offlineVoice = false

    private val permission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) camera.bind() else {
            cameraStatus.text = "Camera permission is needed to record."
            cameraButton.text = if (needsSettings()) "Open camera settings" else "Allow camera"
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val preview = PreviewView(this).apply {
            id = R.id.camera_preview
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FIT_CENTER
            setBackgroundColor(Color.BLACK)
        }
        setContentView(createScreen(preview))
        camera = SignCamera(this, preview, ::cameraStateChanged, ::captureFinished)
        speech = OfflineSpeech(this) { available ->
            offlineVoice = available
            speechStatus.text = if (available) "" else "Install an offline English voice to speak."
        }
        renderClassification(Classification.Unavailable)
        if (classifier.isAvailable) modelStatus.text = "Ready for a sign"
        renderHistory()
    }

    override fun onResume() {
        super.onResume()
        if (hasPermission()) camera.bind()
    }

    override fun onStop() {
        camera.discardInProgress()
        speech.stop()
        super.onStop()
    }

    override fun onDestroy() {
        speech.close()
        worker.execute { classifier.close() }
        worker.shutdown()
        super.onDestroy()
    }

    private fun createScreen(preview: PreviewView): View {
        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(12), dp(20), dp(24))
            isFocusableInTouchMode = true
        }
        val scroll = ScrollView(this).apply {
            isFillViewport = true
            addView(content)
        }
        ViewCompat.setOnApplyWindowInsetsListener(scroll) { view, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout() or
                    WindowInsetsCompat.Type.ime()
            )
            view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }
        content.addView(label("EchoGesture", 30, bold = true))
        content.addView(preview, LinearLayout.LayoutParams(-1, dp(232)).apply { topMargin = dp(12) })
        cameraStatus = label("Allow the camera to record a sign.", 14).apply {
            id = R.id.camera_status
            accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE
        }
        content.addView(cameraStatus)
        cameraButton = button(if (needsSettings()) "Open camera settings" else "Allow camera", ::openCamera)
        recordButton = button("Start recording", ::toggleRecording).apply {
            id = R.id.record_sign
            isEnabled = false
        }
        content.addView(cameraButton)
        content.addView(recordButton)

        content.addView(section("Recognized word"))
        wordView = label("—", 26, bold = true).apply { id = R.id.recognized_word }
        modelStatus = label("", 14).apply {
            id = R.id.recognition_status
            accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE
        }
        addWord = button("Add word to sentence", ::appendWord).apply { id = R.id.add_word }
        content.addView(wordView)
        content.addView(modelStatus)
        content.addView(addWord)

        content.addView(section("Sentence"))
        sentence = EditText(this).apply {
            id = R.id.sentence
            hint = "Type or add recognized words"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES or
                InputType.TYPE_TEXT_FLAG_MULTI_LINE
            minLines = 2
            setText(preferences.getString("draft", ""))
            doAfterTextChanged { preferences.edit().putString("draft", it.toString()).apply() }
        }
        content.addView(sentence)
        content.addView(LinearLayout(this).apply {
            addView(button("Speak", ::speakSentence), LinearLayout.LayoutParams(0, -2, 1f))
            val save = button("Save to history", ::saveSentence).apply { id = R.id.save_sentence }
            addView(save, LinearLayout.LayoutParams(0, -2, 1f))
        })
        speechStatus = label("", 14).apply { accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE }
        content.addView(speechStatus)
        content.addView(section("History"))
        historyView = LinearLayout(this).apply {
            id = R.id.sentence_history
            orientation = LinearLayout.VERTICAL
        }
        content.addView(historyView)
        return scroll
    }

    private fun hasPermission() = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
        PackageManager.PERMISSION_GRANTED

    private fun needsSettings() = !hasPermission() && preferences.getBoolean("cameraAsked", false) &&
        !shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)

    private fun openCamera() {
        when {
            hasPermission() -> camera.bind()
            needsSettings() -> startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:$packageName")))
            else -> {
                preferences.edit().putBoolean("cameraAsked", true).apply()
                permission.launch(Manifest.permission.CAMERA)
            }
        }
    }

    private fun cameraStateChanged(ready: Boolean, message: String) {
        cameraReady = ready
        cameraStatus.text = message
        cameraButton.visibility = if (ready) View.GONE else View.VISIBLE
        cameraButton.text = if (hasPermission()) "Retry camera" else "Allow camera"
        updateRecordButton()
    }

    private fun toggleRecording() {
        if (!hasPermission()) {
            openCamera()
            return
        }
        if (capturing) {
            capturing = false
            busy = true
            cameraStatus.text = "Finishing capture…"
            camera.finish()
        } else if (camera.start()) {
            capturing = true
            pendingWord = null
            addWord.isEnabled = false
            wordView.text = "—"
            modelStatus.text = if (classifier.isAvailable) "Recording…" else "Model unavailable"
            cameraStatus.text = "Recording one sign"
        }
        updateRecordButton()
    }

    private fun captureFinished(outcome: CaptureOutcome) {
        capturing = false
        when (outcome) {
            is CaptureOutcome.Message -> {
                busy = false
                cameraStatus.text = outcome.text
                if (classifier.isAvailable) modelStatus.text = "Ready for a sign"
            }
            is CaptureOutcome.Ready -> {
                cameraStatus.text = "Capture complete"
                busy = classifier.isAvailable
                if (!busy) {
                    outcome.clip.delete()
                    renderClassification(Classification.Unavailable)
                } else {
                    modelStatus.text = "Recognizing sign…"
                    worker.execute {
                        val result = try {
                            classifier.classify(outcome.clip)
                        } catch (error: Exception) {
                            Log.w("EchoGesture", "Local recognition failed", error)
                            Classification.Failed
                        } finally {
                            outcome.clip.delete()
                        }
                        runOnUiThread {
                            if (!isDestroyed) {
                                busy = false
                                renderClassification(result)
                                updateRecordButton()
                            }
                        }
                    }
                }
            }
        }
        updateRecordButton()
    }

    fun renderClassification(result: Classification) {
        val prediction = result as? Classification.Word
        pendingWord = prediction?.gloss?.trim()?.takeIf(String::isNotEmpty)
        wordView.text = pendingWord ?: "—"
        addWord.isEnabled = pendingWord != null
        modelStatus.text = when {
            prediction?.fromTestFixture == true -> "Test fixture"
            result == Classification.Unavailable -> "Model unavailable"
            result == Classification.NoHands -> "No usable hands found. Record the whole sign and try again."
            result == Classification.Failed -> "Could not recognize this recording. Try again."
            pendingWord == null -> "No word recognized"
            else -> ""
        }
    }

    private fun updateRecordButton() {
        recordButton.isEnabled = cameraReady && !busy
        recordButton.text = if (capturing) "Finish recording" else "Start recording"
    }

    private fun appendWord() {
        val word = pendingWord ?: return
        val draft = sentence.text.toString().trimEnd()
        setSentence(if (draft.isEmpty()) word else "$draft $word")
        pendingWord = null
        addWord.isEnabled = false
    }

    private fun setSentence(text: String) {
        sentence.setText(text)
        sentence.setSelection(text.length)
    }

    private fun readHistory(): List<String> {
        val json = JSONArray(preferences.getString("history", "[]") ?: "[]")
        return List(json.length()) { json.getString(it) }
    }

    private fun saveSentence() {
        val text = sentence.text.toString().trim()
        if (text.isEmpty()) {
            speechStatus.text = "Write a sentence before saving."
            return
        }
        val history = (listOf(text) + readHistory().filterNot { it == text }).take(50)
        preferences.edit().putString("history", JSONArray(history).toString()).apply()
        speechStatus.text = "Sentence saved"
        renderHistory()
    }

    private fun renderHistory() {
        historyView.removeAllViews()
        val history = readHistory()
        if (history.isEmpty()) historyView.addView(label("No saved sentences", 14))
        history.forEach { text -> historyView.addView(button(text) { setSentence(text) }) }
    }

    private fun speakSentence() {
        val text = sentence.text.toString().trim()
        speechStatus.text = when {
            text.isEmpty() -> "Write a sentence to speak."
            !offlineVoice -> "Install an offline English voice to speak."
            speech.speak(text) -> ""
            else -> "Speech could not start."
        }
    }

    private fun dp(size: Int) = (size * resources.displayMetrics.density + 0.5f).toInt()

    private fun label(text: String, size: Int, bold: Boolean = false) = TextView(this).apply {
        this.text = text
        textSize = size.toFloat()
        setTextColor(Color.rgb(230, 241, 237))
        if (bold) setTypeface(typeface, Typeface.BOLD)
    }

    private fun section(title: String) = label(title, 16, bold = true).apply {
        setPadding(0, dp(14), 0, dp(4))
        ViewCompat.setAccessibilityHeading(this, true)
    }

    private fun button(title: String, action: () -> Unit) = Button(this).apply {
        text = title
        isAllCaps = false
        minHeight = dp(48)
        backgroundTintList = ColorStateList(
            arrayOf(intArrayOf(-android.R.attr.state_enabled), intArrayOf()),
            intArrayOf(Color.rgb(44, 61, 57), Color.rgb(148, 228, 203)),
        )
        setTextColor(ColorStateList(
            arrayOf(intArrayOf(-android.R.attr.state_enabled), intArrayOf()),
            intArrayOf(Color.rgb(179, 200, 193), Color.rgb(14, 31, 27)),
        ))
        setOnClickListener { action() }
    }
}
