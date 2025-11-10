const uploadButton = document.getElementById("uploadButton");
const inputImage = document.getElementById("inputImage");
const imageChecker = document.getElementById("imageChecker");
const translateButton = document.getElementById("translateButton");
const imageOutput = document.getElementById("imageOutput");
const textOutput = document.getElementById("textOutput");


function adjustTextOutputHeight() {
  //Reset height to auto first to get the natural height
  textOutput.style.height='auto';
  
  //Calculate the required height with a minimum
  const scrollHeight =textOutput.scrollHeight;
  const minHeight = 150; // Minimum height in pixels
  const calculatedHeight = Math.max(minHeight, scrollHeight + 20); // Padding 

  //Set the new height 
  textOutput.style.height = calculatedHeight + 'px';
}
//temporary image
let selectedFile = null;

inputImage.addEventListener("change", () => {
  selectedFile = inputImage.files[0];
  
  if (selectedFile) {
    // change text in image checker to confirm file uploaded
    imageOutput.innerHTML = `Image uploaded: ${selectedFile.name}`;
  } else {
    // go back to normal text if the file is invalid
    imageChecker.textContent = "No Image Has Been Uploaded";
    imageChecker.style.color = "red";
  }
});
//upload Button
uploadButton.addEventListener("click", () => {
  inputImage.click();
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
}


);