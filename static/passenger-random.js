// 乘客欄位隨機產生：新增乘客頁與生存預測頁共用同一份邏輯（DRY）。
//
// 用法：在按鈕加 data-random="name|ticket|cabin"，並置於目標表單內即可；
// 透過事件委派自動處理，免逐一綁定。name 會依同表單的 Sex 決定稱謂。
(function () {
    "use strict";

    const SURNAMES = [
        "Braund", "Cumings", "Heikkinen", "Futrelle", "Allen", "Moran",
        "McCarthy", "Palsson", "Johnson", "Nasser", "Sandstrom", "Bonnell",
        "Andersson", "Hewlett", "Rice", "Williams", "Vander Planke", "Masselmani",
        "Fortune", "Beesley", "Sloper", "Asplund", "Harper", "Goodwin",
    ];
    const MALE_FIRST = [
        "Owen Harris", "William Henry", "James", "Timothy J", "Gosta Leonard",
        "Anders Johan", "Charles Eugene", "William Thompson", "Edward H", "Lawrence",
    ];
    const FEMALE_FIRST = [
        "Laina", "Lily May", "Elisabeth Vilhelmina", "Hulda Amanda", "Anna",
        "Margaret Norton", "Marguerite Rut", "Torborg Danira", "Fatima", "Adele Kiamie",
    ];
    const MALE_TITLES = ["Mr", "Master"];
    const FEMALE_TITLES = ["Mrs", "Miss"];
    const TICKET_PREFIXES = ["", "", "", "A/5", "PC", "STON/O2.", "C.A.", "SC/Paris", "W./C."];
    const DECKS = ["A", "B", "C", "D", "E", "F", "G"];

    function randomItem(items) {
        return items[Math.floor(Math.random() * items.length)];
    }

    // 姓名格式對齊資料集「Surname, Title. First」，稱謂依 Sex 決定，
    // 使萃取的 Title 與 Sex 一致（呼應特徵工程的 Title 規則）。
    function generateName(isFemale) {
        const title = randomItem(isFemale ? FEMALE_TITLES : MALE_TITLES);
        const first = randomItem(isFemale ? FEMALE_FIRST : MALE_FIRST);
        return randomItem(SURNAMES) + ", " + title + ". " + first;
    }

    // 票號：部分帶資料集常見前綴（A/5、PC…），其餘純數字。
    function generateTicket() {
        const prefix = randomItem(TICKET_PREFIXES);
        const number = 1000 + Math.floor(Math.random() * 999000);
        return prefix ? prefix + " " + number : String(number);
    }

    // 艙房：甲板字母 A–G + 房號。
    function generateCabin() {
        return randomItem(DECKS) + (1 + Math.floor(Math.random() * 130));
    }

    function setValue(form, name, value) {
        const input = form.querySelector('[name="' + name + '"]');
        if (input) {
            input.value = value;
        }
    }

    function fillRandom(button) {
        const form = button.closest("form");
        if (!form) {
            return;
        }
        const field = button.dataset.random;
        if (field === "name") {
            const sex = form.querySelector('[name="Sex"]');
            setValue(form, "Name", generateName(sex && sex.value === "female"));
        } else if (field === "ticket") {
            setValue(form, "Ticket", generateTicket());
        } else if (field === "cabin") {
            setValue(form, "Cabin", generateCabin());
        }
    }

    // 事件委派：任何帶 data-random 的按鈕都能觸發。
    document.addEventListener("click", function (event) {
        const button = event.target.closest("[data-random]");
        if (button) {
            event.preventDefault();
            fillRandom(button);
        }
    });
})();
