// lib/api.ts
export async function fetchEEGData() {
    const response = await fetch("http://127.0.0.1:5020/record_predict", {
      method: "POST", // Use POST method
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({}), // Include an empty body or any required data
    });
  
    if (!response.ok) {
      throw new Error("Failed to fetch EEG data");
    }
  
    return response.json();
  }

export async function fetchJsonData() {
    const response = await fetch("http://127.0.0.1:5020/get_json", {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });
  
    if (!response.ok) {
      throw new Error("Failed to fetch JSON data");
    }
  
    return response.json();
  }

