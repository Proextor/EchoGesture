package com.echogesture.app

internal object LegacyFeatures {
    const val FRAMES = 64
    const val POINTS = 54
    const val RAW_SIZE = POINTS * 3
    const val FEATURE_SIZE = POINTS * 9
    val poseIds = intArrayOf(11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
    private val handParents = intArrayOf(0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19)
    private val parents = handParents + handParents.map { it + 21 }.toIntArray() +
        intArrayOf(43, 42, 42, 43, 44, 45, 42, 43, 48, 49, 50, 51)

    fun sampleIndices(frameCount: Int): IntArray {
        require(frameCount > 0)
        val step = (frameCount - 1).toDouble() / (FRAMES - 1)
        return IntArray(FRAMES) { Math.rint(it * step).toInt() }
    }

    fun present(raw: FloatArray, offset: Int): Boolean =
        raw[offset] != 0f || raw[offset + 1] != 0f || raw[offset + 2] != 0f

    fun normalize(row: FloatArray) {
        val left = 42 * 3
        val right = 43 * 3
        if (!present(row, left) && !present(row, right)) return
        val center = FloatArray(3) { (row[left + it] + row[right + it]) * 0.5f }
        val dx = row[left] - row[right]
        val dy = row[left + 1] - row[right + 1]
        val scale = kotlin.math.sqrt(dx * dx + dy * dy).coerceAtLeast(1e-3f)
        for (point in 0 until POINTS) {
            val offset = point * 3
            if (!present(row, offset)) continue
            for (axis in 0..2) row[offset + axis] = (row[offset + axis] - center[axis]) / scale
        }
    }

    fun engineer(raw: FloatArray): FloatArray {
        require(raw.size == FRAMES * RAW_SIZE && raw.all(Float::isFinite))
        val features = FloatArray(FRAMES * FEATURE_SIZE)
        for (frame in 0 until FRAMES) {
            val row = frame * RAW_SIZE
            for (point in 0 until POINTS) {
                val input = row + point * 3
                val parent = row + parents[point] * 3
                val output = frame * FEATURE_SIZE + point * 9
                val validBone = present(raw, input) && present(raw, parent)
                for (axis in 0..2) {
                    val position = raw[input + axis]
                    features[output + axis] = position
                    if (frame > 0) features[output + 3 + axis] = position - raw[input + axis - RAW_SIZE]
                    if (validBone) features[output + 6 + axis] = position - raw[parent + axis]
                }
            }
        }
        return features
    }
}
