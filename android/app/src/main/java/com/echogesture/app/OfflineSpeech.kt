package com.echogesture.app

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech

class OfflineSpeech(context: Context, onAvailability: (Boolean) -> Unit) {
    private var closed = false
    private var ready = false
    private val handler = Handler(Looper.getMainLooper())
    private val engine: TextToSpeech = TextToSpeech(context.applicationContext) { status ->
        handler.post {
            if (closed) return@post
            ready = status == TextToSpeech.SUCCESS && selectOfflineVoice()
            onAvailability(ready)
        }
    }

    private fun selectOfflineVoice(): Boolean {
        val offline = engine.voices.orEmpty().filter { voice ->
            voice.locale.language == "en" && !voice.isNetworkConnectionRequired &&
                TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED !in voice.features.orEmpty()
        }
        val preferred = offline.firstOrNull { it == engine.defaultVoice } ?: offline.firstOrNull()
        return preferred != null && engine.setVoice(preferred) == TextToSpeech.SUCCESS
    }

    fun speak(text: String): Boolean =
        ready && text.length <= TextToSpeech.getMaxSpeechInputLength() &&
            engine.speak(text, TextToSpeech.QUEUE_FLUSH, null, "sentence") == TextToSpeech.SUCCESS

    fun stop() {
        engine.stop()
    }

    fun close() {
        closed = true
        engine.stop()
        engine.shutdown()
    }
}
