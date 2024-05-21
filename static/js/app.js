// Get the form element
const recommendForm = document.querySelector('form');

// Add an event listener for form submission
recommendForm.addEventListener('submit', function(event) {
  // Get the selected tag value
  const selectedTag = document.querySelector('#tag').value;

  // Check if a tag is selected
  if (!selectedTag) {
    // If no tag is selected, prevent form submission and display an error message
    event.preventDefault();
    alert('Please select a tag before submitting the form.');
  }
});
