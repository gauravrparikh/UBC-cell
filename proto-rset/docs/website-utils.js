//https://www.w3schools.com/howto/howto_js_copy_clipboard.asp

window.onload = () => {
    //your code to run here

    var protoButton = document.getElementById('protopnet-button');
    protoButton.addEventListener('click', function(){
        copyToClipboard('protopnet');
});

    var deformableButton = document.getElementById('deformable-button');
        deformableButton.addEventListener('click', function(){
            copyToClipboard('deformable');
    });
}


function copyToClipboard(paperType) {
    // Get the text field
    var copyText = document.getElementById(paperType + '-text');
  
    // Select the text field
    //copyText.select();
    //console.log(typeof(copyText))
    //copyText.setSelectionRange(0, 99999); // For mobile devices
    textContent = copyText.textContent;

     // Copy the text inside the text field
    navigator.clipboard.writeText(textContent);
  }