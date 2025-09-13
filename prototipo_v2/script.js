document.addEventListener('DOMContentLoaded', () => {
    // Tab switching logic
    const tabs = document.querySelectorAll('.tab-link');
    const tabContents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const target = document.getElementById(tab.dataset.tab);

            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            tabContents.forEach(content => content.classList.remove('active'));
            target.classList.add('active');
        });
    });

    // Combobox logic
    const comboboxInput = document.getElementById('material-input');
    const comboboxOptionsContainer = document.querySelector('.combobox-options');
    const comboboxOptions = document.querySelectorAll('.combobox-option');

    comboboxInput.addEventListener('focus', () => {
        comboboxOptionsContainer.style.display = 'block';
    });

    comboboxInput.addEventListener('blur', () => {
        // Delay hiding to allow click on option
        setTimeout(() => {
            comboboxOptionsContainer.style.display = 'none';
        }, 200);
    });

    comboboxInput.addEventListener('input', () => {
        const filter = comboboxInput.value.toUpperCase();
        comboboxOptions.forEach(option => {
            const value = option.textContent || option.innerText;
            if (value.toUpperCase().indexOf(filter) > -1) {
                option.style.display = '';
            } else {
                option.style.display = 'none';
            }
        });
    });

    comboboxOptions.forEach(option => {
        option.addEventListener('mousedown', (e) => {
            e.preventDefault();
            comboboxInput.value = option.dataset.value;
            comboboxOptionsContainer.style.display = 'none';
        });
    });

    // Modal logic
    const modal = document.getElementById('suggest-material-modal');
    const modalTitle = document.getElementById('modal-title');
    const closeModalBtn = document.querySelector('.close-button');
    const cancelModalBtn = document.getElementById('cancel-btn');
    const useApuBtn = document.getElementById('use-apu-btn');

    const openModal = (materialName) => {
        modalTitle.textContent = `Sugerir Nuevo Material: "${materialName}"`;
        modal.style.display = 'block';
    };

    const closeModal = () => {
        modal.style.display = 'none';
    };

    closeModalBtn.addEventListener('click', closeModal);
    cancelModalBtn.addEventListener('click', closeModal);
    window.addEventListener('click', (event) => {
        if (event.target == modal) {
            closeModal();
        }
    });

    comboboxInput.addEventListener('blur', () => {
        setTimeout(() => { // Allow mousedown on option to fire first
            const inputValue = comboboxInput.value.trim();
            if (inputValue === '') return;

            const existingOptions = Array.from(comboboxOptions).map(opt => opt.dataset.value);
            if (!existingOptions.includes(inputValue)) {
                openModal(inputValue);
            }
        }, 200);
    });

    // Estimation simulation logic
    const resultsOutput = document.getElementById('results-output');
    const quantityInput = document.getElementById('quantity');

    const simulateEstimation = () => {
        const quantity = parseFloat(quantityInput.value) || 100; // Default to 100 if empty
        const material = comboboxInput.value;
        const apuBase = document.getElementById('apu-base-select').selectedOptions[0].text;

        const estimatedCost = (Math.random() * 50 + 80).toFixed(2); // Random cost between 80-130
        const total = (quantity * estimatedCost).toFixed(2);

        resultsOutput.innerHTML = `
            <p><strong>Material:</strong> ${material}</p>
            <p><strong>APU Base Utilizado:</strong> ${apuBase}</p>
            <p><strong>Cantidad:</strong> ${quantity} m2</p>
            <p><strong>Costo Estimado por m2:</strong> $${estimatedCost}</p>
            <hr>
            <p><strong>Total Estimado:</strong> $${total}</p>
        `;
        closeModal();
    };

    useApuBtn.addEventListener('click', simulateEstimation);
    document.getElementById('estimate-btn').addEventListener('click', () => {
        // Also allow direct estimation if material exists
        const inputValue = comboboxInput.value.trim();
        const existingOptions = Array.from(comboboxOptions).map(opt => opt.dataset.value);
        if (inputValue !== '' && existingOptions.includes(inputValue)) {
             // A simplified simulation for existing materials
            const quantity = parseFloat(quantityInput.value) || 100;
            const estimatedCost = (Math.random() * 40 + 70).toFixed(2); // 70-110
            const total = (quantity * estimatedCost).toFixed(2);
            resultsOutput.innerHTML = `
                <p><strong>Material:</strong> ${inputValue}</p>
                <p><strong>Cantidad:</strong> ${quantity} m2</p>
                <p><strong>Costo Estimado por m2:</strong> $${estimatedCost}</p>
                <hr>
                <p><strong>Total Estimado:</strong> $${total}</p>
            `;
        } else if(inputValue !== '') {
            openModal(inputValue);
        }
    });
});
