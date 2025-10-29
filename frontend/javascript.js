const uploadButton = document.getElementById("uploadButton");
const inputImage = document.getElementById("inputImage");
const imageChecker = document.getElementById("imageChecker");
const translateButton = document.getElementById("translateButton");
const imageOutput = document.getElementById("imageOutput");
const textOutput = document.getElementById("textOutput");

//temporary image
let selectedFile = null;

//file input
uploadButton.addEventListener("click", () => inputImage.click());

inputImage.addEventListener("change", () => {
  selectedFile = inputImage.files[0];
  
  if (selectedFile) {
    // change text in image checker to confirm file uploaded
    imageChecker.textContent = `Image uploaded: ${selectedFile.name}`;
    imageChecker.style.color = "green"; // optional: visual cue
  } else {
    // go back to normal text if the file is invalid
    imageChecker.textContent = "No Image Has Been Uploaded";
    imageChecker.style.color = "red";
  }
});
//translate now buttom
translateButton.addEventListener("click", async () => {
  if(!selectedFile) {
     alert("Upload an image first!");
     return; 
  }

  // indicate translating state
  const originalButtonText = translateButton.textContent;
  translateButton.disabled = true;
  translateButton.textContent = 'Translating... please wait';
  imageOutput.innerHTML = `<p>Translating... please wait</p>`;

  //stuff to ready things to send to AI
  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    //send image to AI
    const response = await fetch("/process_image", {
    method: "POST",
    body: formData,
    });

    if (!response.ok) {
    throw new Error(`Server error: ${response.status}`);
    }

    // parsing
    const data = await response.json();

    // update outputs: image and textual predictions
    imageOutput.innerHTML = `<img src="${data.output_url}" alt="Processed Baybayin Image">`;
    // show best translation and full predictions text (predictions_txt)
    const predText = data.predictions_txt ? data.predictions_txt : '';
    textOutput.innerHTML = `<strong>Translation:</strong> ${data.translation}<br><pre style="white-space:pre-wrap; text-align:left;">${predText}</pre>`;
  } catch (error) {
    console.error("Error:", error);
    imageOutput.innerHTML = "<p>Error processing image.</p>";
    textOutput.innerHTML = "<p>Translation failed.</p>";
  } finally {
    // restore button state
    translateButton.disabled = false;
    translateButton.textContent = originalButtonText;
  }
});