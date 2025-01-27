document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('upload-form');
    const dropZone = document.querySelector('.drop-zone');
    const input = dropZone.querySelector('input');
    
    ['dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    dropZone.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        input.files = files;
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(form);
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.error) {
                alert(data.error);
                return;
            }
            
            document.getElementById('original-image').src = data.original;
            document.getElementById('processed-image').src = data.processed;
            document.getElementById('result').classList.remove('hidden');
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while processing the image');
        }
    });
});