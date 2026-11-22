package com.echogesture.app

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test

class CaptureTest {
    @Test
    fun requiresBytesAndRecordedTime() {
        assertEquals(CaptureValidity.EMPTY, captureValidity(0, 500_000_000))
        assertEquals(CaptureValidity.EMPTY, captureValidity(100, 0))
    }

    @Test
    fun enforcesTheMinimumDuration() {
        assertEquals(CaptureValidity.TOO_SHORT, captureValidity(100, 349_999_999))
        assertEquals(CaptureValidity.READY, captureValidity(100, 350_000_000))
    }

    @Test
    fun unavailableClassifierReturnsNoWord() {
        assertFalse(UnavailableClassifier.isAvailable)
        assertEquals(Classification.Unavailable, UnavailableClassifier.classify(File("unused.mp4")))
    }
}
