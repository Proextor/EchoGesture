import java.security.MessageDigest

plugins {
    id("com.android.application") version "8.7.3"
    id("org.jetbrains.kotlin.android") version "2.1.21"
}

android {
    namespace = "com.echogesture.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.echogesture.app"
        minSdk = 24
        targetSdk = 35
        versionCode = 2
        versionName = "0.2-five-seed"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    sourceSets["main"].assets.srcDir(layout.buildDirectory.dir("generated/recognitionAssets"))
    sourceSets["androidTest"].assets.srcDir(rootProject.file("../data/five-seed/fixtures"))
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    implementation("androidx.activity:activity-ktx:1.10.1")
    implementation("androidx.camera:camera-camera2:1.4.2")
    implementation("androidx.camera:camera-lifecycle:1.4.2")
    implementation("androidx.camera:camera-video:1.4.2")
    implementation("androidx.camera:camera-view:1.4.2")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.32.0")
    implementation("com.google.mediapipe:tasks-vision:1.2.3")
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test:core:1.6.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
}

val repository = rootProject.projectDir.parentFile
val handoff = repository.resolve("data/five-seed")
val seedModels = (1..5).map { handoff.resolve("seed_$it.onnx") }
val holisticModel = repository.resolve("data/android-parity/holistic_landmarker.task")
val vocabulary = handoff.resolve("vocab.json")
val stageRecognitionAssets by tasks.registering(Sync::class) {
    from(seedModels, vocabulary, holisticModel)
    into(layout.buildDirectory.dir("generated/recognitionAssets"))
    doFirst {
        for (asset in seedModels + vocabulary) check(asset.isFile) { "Missing handoff asset: $asset" }
        check(holisticModel.isFile) { "Prepare data/android-parity/holistic_landmarker.task as described in android/README.md." }
        val hash = MessageDigest.getInstance("SHA-256").digest(holisticModel.readBytes())
            .joinToString("") { "%02x".format(it) }
        check(hash == "e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8") {
            "The Holistic model does not match the documented version."
        }
    }
}
tasks.configureEach {
    if (name == "preBuild") dependsOn(stageRecognitionAssets)
}
