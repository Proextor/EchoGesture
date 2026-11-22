package com.echogesture.app

import java.io.File

interface SignClassifier : AutoCloseable {
    val isAvailable: Boolean

    /** Reads a finalized, silent, app-private MP4 on a background thread. */
    fun classify(clip: File): Classification
    override fun close() = Unit
}

sealed interface Classification {
    data class Word(val gloss: String, val fromTestFixture: Boolean = false) : Classification
    data object Unavailable : Classification
    data object NoHands : Classification
    data object Failed : Classification
}

object UnavailableClassifier : SignClassifier {
    override val isAvailable = false
    override fun classify(clip: File) = Classification.Unavailable
}

enum class CaptureValidity { EMPTY, TOO_SHORT, READY }

fun captureValidity(bytes: Long, durationNanos: Long): CaptureValidity = when {
    bytes <= 0 || durationNanos <= 0 -> CaptureValidity.EMPTY
    durationNanos < 350_000_000L -> CaptureValidity.TOO_SHORT
    else -> CaptureValidity.READY
}
