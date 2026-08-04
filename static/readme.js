const THEME_STORAGE_KEY = "clock-theme";
const themeToggleBtn = document.getElementById("themeToggleBtn");

function applyTheme(theme) {
  if (theme === "light") {
    document.body.classList.add("light-theme");
  } else {
    document.body.classList.remove("light-theme");
  }

  if (themeToggleBtn) {
    themeToggleBtn.title = theme === "light" ? "Switch to dark theme" : "Switch to light theme";
  }
}

function initializeTheme() {
  const savedTheme = localStorage.getItem(THEME_STORAGE_KEY);
  if (savedTheme === "light" || savedTheme === "dark") {
    applyTheme(savedTheme);
    return;
  }

  localStorage.setItem(THEME_STORAGE_KEY, "dark");
  applyTheme("dark");
}

function toggleTheme() {
  const current = document.body.classList.contains("light-theme") ? "light" : "dark";
  const next = current === "light" ? "dark" : "light";
  localStorage.setItem(THEME_STORAGE_KEY, next);
  applyTheme(next);
}

if (themeToggleBtn) {
  themeToggleBtn.addEventListener("click", () => {
    toggleTheme();
  });
}

initializeTheme();
