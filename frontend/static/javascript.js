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

    // parse JSON response even when not OK so we can surface server-side error details
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      const serverMsg = data && (data.error || data.details) ? `${data.error || 'Server error'}: ${data.details || ''}` : `Server error: ${response.status}`;
      throw new Error(serverMsg);
    }

    // update outputs: image and textual predictions
    // Insert an actual <img> element (safer than innerHTML) and constrain its size
    imageOutput.innerHTML = '';
    if (data && data.output_url) {
      const img = document.createElement('img');
      img.src = data.output_url;
      img.alt = 'Processed Baybayin Image';
      img.className = 'processed-image';
      // accessibility: announce load
      img.setAttribute('role', 'img');
      // apply inline constraints as a fallback
      img.style.maxWidth = '100%';
      img.style.maxHeight = '50vh';
      img.style.objectFit = 'contain';
      img.style.display = 'block';
      img.style.margin = '0 auto';
      imageOutput.appendChild(img);
    } else {
      imageOutput.innerHTML = '<p>No image returned from server.</p>';
    }
    // show best translation, english translation (if available), and full predictions text (predictions_txt)
    const predText = data && data.predictions_txt ? data.predictions_txt : '';
    let reconHtml = `<strong>Translation:</strong> ${data.translation || ''}`;
    if (data && data.translation_en) {
      reconHtml += `<br><em style="font-size:0.9em; color:#555;"><strong>English:</strong> ${data.translation_en}</em>`;
    }
    textOutput.innerHTML = reconHtml + `<br><pre style="white-space:pre-wrap; text-align:left;">${predText}</pre>`;
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