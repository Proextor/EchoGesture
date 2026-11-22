package com.echogesture.app

import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test
import org.junit.runner.RunWith

private object WordFixture : SignClassifier {
    override val isAvailable = true
    override fun classify(clip: File) = Classification.Word("THANK YOU", fromTestFixture = true)
}

@RunWith(AndroidJUnit4::class)
class SentenceUiTest {
    @Test
    fun fixtureWordAppendsToEditableSentence() {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                val status = activity.findViewById<TextView>(R.id.recognition_status)
                val word = activity.findViewById<TextView>(R.id.recognized_word)
                val sentence = activity.findViewById<EditText>(R.id.sentence)
                val add = activity.findViewById<Button>(R.id.add_word)
                assertEquals(if (android.os.Build.VERSION.SDK_INT >= 28) "Ready for a sign" else "Model unavailable",
                    status.text.toString())
                assertFalse(add.isEnabled)
                sentence.setText("I sign")
                activity.renderClassification(WordFixture.classify(File("unused.mp4")))
                assertEquals("Test fixture", status.text.toString())
                assertEquals("THANK YOU", word.text.toString())
                add.performClick()
                assertEquals("I sign THANK YOU", sentence.text.toString())
                assertFalse(add.isEnabled)
            }
        }
    }

    @Test
    fun editedSentenceCanBeRestoredFromLocalHistory() {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                activity.findViewById<EditText>(R.id.sentence).setText("Saved offline")
                activity.findViewById<Button>(R.id.save_sentence).performClick()
                activity.findViewById<EditText>(R.id.sentence).setText("Different draft")
            }
            scenario.recreate()
            scenario.onActivity { activity ->
                val sentence = activity.findViewById<EditText>(R.id.sentence)
                val history = activity.findViewById<LinearLayout>(R.id.sentence_history)
                assertEquals("Different draft", sentence.text.toString())
                history.getChildAt(0).performClick()
                assertEquals("Saved offline", sentence.text.toString())
            }
        }
    }
}
