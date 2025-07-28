const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("file");
const uploadStatus = document.getElementById("uploadStatus");

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];

  if (!file) {
    uploadStatus.innerHTML = "⚠️ Please select a file.";
    return;
  }

  uploadStatus.innerHTML = "⏳ Uploading...";

  const reader = new FileReader();
  reader.onload = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/Upload_files", {
        method: "POST",
        headers: {
          "Content-Type": "application/pdf",
          "encoded-filename": encodeURIComponent(file.name)  
        },
        body: reader.result
      });

      const result = await response.json();

      if (response.ok) {
        uploadStatus.innerHTML = `✅ <strong>${result.message}</strong><br>📌 Chunks Added: ${result.chunks_added}`;
      } else {
        uploadStatus.innerHTML = `❌ Error: ${result.detail}`;
      }
    } catch (err) {
      uploadStatus.innerHTML = `❌ Upload failed: ${err.message}`;
    }
  };
  reader.readAsArrayBuffer(file);
});

// Handle Query Submission
const queryForm = document.getElementById("queryForm");
const questionInput = document.getElementById("question");
const answerOutput = document.getElementById("answerOutput");
const submissionStatus = document.getElementById("submissionStatus");

queryForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const question = questionInput.value.trim();
  if (!question) {
    submissionStatus.innerHTML = "⚠️ Please enter a question.";
    return;
  }

  submissionStatus.innerHTML = "⏳ Sending question...";
  answerOutput.innerHTML = "";

  try {
    const response = await fetch(`http://127.0.0.1:8000/answer?query=${encodeURIComponent(question)}`, {
      method: "GET", // ✅ GET method — no body allowed
    });

    const answer = await response.text(); // ✅ Server returns HTML content

    if (response.ok) {
      submissionStatus.innerHTML = "✅ Successfully submitted";
      answerOutput.innerHTML = `💡 <strong>Answer:</strong> ${answer}`;
    } else {
      submissionStatus.innerHTML = "❌ Server error.";
      answerOutput.innerHTML = `❌ ${answer}`;
    }
  } catch (err) {
    submissionStatus.innerHTML = "❌ Network error.";
    answerOutput.innerHTML = `❌ ${err.message}`;
  }
});
