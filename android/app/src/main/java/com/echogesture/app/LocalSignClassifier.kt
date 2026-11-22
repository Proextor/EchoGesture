package com.echogesture.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.Build
import org.json.JSONArray
import java.io.File
import java.nio.FloatBuffer
import java.util.Locale
import kotlin.math.exp

internal data class SignInference(
    val features: FloatArray,
    val logits: List<FloatArray>,
    val combined: DoubleArray,
    val topFive: List<String>,
    val confidence: Double,
)

internal class LocalSignClassifier(context: Context) : SignClassifier {
    private val app = context.applicationContext
    override val isAvailable = Build.VERSION.SDK_INT >= 28
    private var model: LegacyOnnxModel? = null

    override fun classify(clip: File): Classification {
        if (Build.VERSION.SDK_INT < 28) return Classification.Unavailable
        val landmarks = HolisticVideo(app).extract(clip)
        if (!landmarks.hasHands) return Classification.NoHands
        val result = inferKeypoints(landmarks.raw)
        return Classification.Word(result.topFive.first())
    }

    internal fun inferKeypoints(raw: FloatArray): SignInference {
        val activeModel = model ?: LegacyOnnxModel(app).also { model = it }
        return activeModel.infer(raw)
    }

    override fun close() {
        model?.close()
        model = null
    }
}

private class LegacyOnnxModel(context: Context) : AutoCloseable {
    private val environment = OrtEnvironment.getEnvironment()
    private val assets = context.assets
    private val labels = assets.open("vocab.json").bufferedReader().use { reader ->
        val names = JSONArray(reader.readText())
        List(names.length()) { names.getString(it).uppercase(Locale.ROOT) }
    }.also { require(it.size == 100 && it.toSet().size == it.size) }

    private val sessions = mutableListOf<OrtSession>()

    init {
        val names = (1..5).map { "seed_$it.onnx" }
        try {
            OrtSession.SessionOptions().use { options ->
                options.setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(1, 4))
                options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
                for (name in names) {
                    val bytes = assets.open(name).use { it.readBytes() }
                    sessions.add(environment.createSession(bytes, options))
                }
            }
        } catch (error: Exception) {
            close()
            throw error
        }
    }

    fun infer(raw: FloatArray): SignInference {
        val features = LegacyFeatures.engineer(raw)
        val shape = longArrayOf(1, LegacyFeatures.FRAMES.toLong(), LegacyFeatures.FEATURE_SIZE.toLong())
        val logits = OnnxTensor.createTensor(environment, FloatBuffer.wrap(features), shape).use { tensor ->
            sessions.map { session ->
                session.run(mapOf(session.inputNames.single() to tensor)).use { result ->
                    val scores = (result[0].value as Array<*>).single() as FloatArray
                    require(scores.size == labels.size && scores.all(Float::isFinite))
                    scores.copyOf()
                }
            }
        }
        val combined = DoubleArray(labels.size)
        val weight = 1.0 / sessions.size
        for (scores in logits) for (index in scores.indices) combined[index] += scores[index] * weight
        val peak = combined.max()
        val denominator = combined.sumOf { exp(it - peak) }
        // Use the same float64 softmax as the Python wrapper.
        val probabilities = DoubleArray(labels.size) { exp(combined[it] - peak) / denominator }
        val topFive = probabilities.indices.sortedByDescending { probabilities[it] }.take(5)
        return SignInference(features, logits, combined, topFive.map(labels::get), probabilities[topFive.first()])
    }

    override fun close() {
        sessions.forEach(OrtSession::close)
        sessions.clear()
    }
}
