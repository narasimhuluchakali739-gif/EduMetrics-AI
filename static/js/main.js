/**
 * EDUMETRICS AI — Main Application Client Scripts
 * Provides theme management, dynamic prediction interactions, input synchronization,
 * toast notifications, modal dialogs, and Chart.js initialization helpers.
 */

// 1. Theme Management (Light / Dark Mode)
(function initTheme() {
  const savedTheme = localStorage.getItem('edumetrics-theme') || 
    (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  document.documentElement.setAttribute('data-theme', savedTheme);
})();

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'light';
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('edumetrics-theme', next);
  updateThemeIcon(next);
}

function updateThemeIcon(theme) {
  const btn = document.getElementById('theme-toggle-btn');
  if (!btn) return;
  btn.innerHTML = theme === 'dark' 
    ? '<i data-lucide="sun" style="width:18px;height:18px;"></i>' 
    : '<i data-lucide="moon" style="width:18px;height:18px;"></i>';
  if (window.lucide) lucide.createIcons();
}

// 2. Toast Notification System
function showToast(message, type = 'info', duration = 4000) {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    document.body.appendChild(container);
  }

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  
  let iconName = 'info';
  if (type === 'success') iconName = 'check-circle';
  if (type === 'error') iconName = 'alert-triangle';

  toast.innerHTML = `
    <i data-lucide="${iconName}" style="width:20px;height:20px;flex-shrink:0;"></i>
    <span style="flex:1;">${message}</span>
    <button type="button" style="background:none;border:none;cursor:pointer;color:inherit;" onclick="this.parentElement.remove()">
      <i data-lucide="x" style="width:16px;height:16px;"></i>
    </button>
  `;

  container.appendChild(toast);
  if (window.lucide) lucide.createIcons();

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(100%)';
    setTimeout(() => toast.remove(), 250);
  }, duration);
}

// 3. Modal Manager
let activeModalAction = null;

function openConfirmModal(title, message, confirmBtnText, onConfirmCallback) {
  const modal = document.getElementById('confirm-modal');
  if (!modal) return;
  document.getElementById('modal-title').textContent = title;
  document.getElementById('modal-message').textContent = message;
  const btn = document.getElementById('modal-confirm-btn');
  btn.textContent = confirmBtnText;
  activeModalAction = onConfirmCallback;
  modal.classList.add('active');
}

function closeConfirmModal() {
  const modal = document.getElementById('confirm-modal');
  if (modal) modal.classList.remove('active');
  activeModalAction = null;
}

function executeModalConfirm() {
  if (typeof activeModalAction === 'function') {
    activeModalAction();
  }
  closeConfirmModal();
}

// 4. Form Synchronizer (Sliders & Number Inputs)
function syncInputPair(sliderId, numberId) {
  const slider = document.getElementById(sliderId);
  const number = document.getElementById(numberId);
  if (!slider || !number) return;

  slider.addEventListener('input', () => {
    number.value = slider.value;
  });

  number.addEventListener('input', () => {
    const val = parseFloat(number.value);
    if (!isNaN(val)) {
      slider.value = val;
    }
  });
}

// 5. Preset Profiles for Prediction Studio
const PRESET_PROFILES = {
  high_achiever: {
    student_id: 'STU-TOP-901',
    student_name: 'Aiden Vance',
    study_hours: 34.0,
    attendance: 96.0,
    previous_score: 91.5,
    assignment_score: 98.0,
    sleep_hours: 7.5,
    tutoring_sessions: 3,
  },
  balanced: {
    student_id: 'STU-MED-442',
    student_name: 'Maya Patel',
    study_hours: 22.0,
    attendance: 84.0,
    previous_score: 75.0,
    assignment_score: 82.0,
    sleep_hours: 7.5,
    tutoring_sessions: 2,
  },
  borderline: {
    student_id: 'STU-BDR-108',
    student_name: 'Liam Cooper',
    study_hours: 14.0,
    attendance: 72.0,
    previous_score: 55.0,
    assignment_score: 64.0,
    sleep_hours: 6.0,
    tutoring_sessions: 0,
  },
  at_risk: {
    student_id: 'STU-RSK-055',
    student_name: 'Jordan Brooks',
    study_hours: 8.0,
    attendance: 58.0,
    previous_score: 42.0,
    assignment_score: 48.0,
    sleep_hours: 5.0,
    tutoring_sessions: 0,
  },
};

function applyPresetProfile(presetKey) {
  const profile = PRESET_PROFILES[presetKey];
  if (!profile) return;

  for (const [key, val] of Object.entries(profile)) {
    const el = document.getElementById(key);
    const slider = document.getElementById(`${key}_slider`);
    if (el) el.value = val;
    if (slider) slider.value = val;
  }
  showToast(`Applied preset: ${presetKey.replace('_', ' ').toUpperCase()}`, 'info', 2000);
}

// 6. Delete Record Handler
function deletePredictionRecord(recordId) {
  openConfirmModal(
    'Delete Record',
    'Are you sure you want to permanently delete this student prediction record? This action cannot be undone.',
    'Delete Permanently',
    async () => {
      try {
        const res = await fetch(`/api/history/${recordId}`, { method: 'DELETE' });
        const data = await res.json();
        if (data.success) {
          showToast('Record deleted successfully', 'success');
          const row = document.getElementById(`record-row-${recordId}`);
          if (row) {
            row.style.opacity = '0';
            setTimeout(() => row.remove(), 250);
          } else {
            setTimeout(() => window.location.reload(), 400);
          }
        } else {
          showToast(data.error || 'Failed to delete record', 'error');
        }
      } catch (err) {
        showToast('Network error while deleting record', 'error');
      }
    }
  );
}

// 7. Mobile Navigation Toggle
function toggleMobileNav() {
  const drawer = document.getElementById('mobile-drawer');
  if (drawer) {
    drawer.classList.toggle('active');
  }
}

// Document Ready Initialization
document.addEventListener('DOMContentLoaded', () => {
  if (window.lucide) {
    lucide.createIcons();
  }
  const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
  updateThemeIcon(currentTheme);

  // Sync sliders on predict form if present
  const features = ['study_hours', 'attendance', 'previous_score', 'assignment_score', 'sleep_hours', 'tutoring_sessions'];
  features.forEach(feat => syncInputPair(`${feat}_slider`, feat));
});
