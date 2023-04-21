#include "inference_config.h"

// Function to remove spaces from a string
void removeSpaces(std::string &str) {
  str.erase(std::remove(str.begin(), str.end(), ' '), str.end());
}

std::set<int> setFromList(std::string &str) {
  std::string stringified_list(str);
  std::set<int> int_set;

  // Remove square brackets from the string
  stringified_list.erase(
      std::remove(stringified_list.begin(), stringified_list.end(), '['),
      stringified_list.end());
  stringified_list.erase(
      std::remove(stringified_list.begin(), stringified_list.end(), ']'),
      stringified_list.end());

  // Use stringstream to parse the comma-separated list of integers
  std::stringstream ss(stringified_list);
  std::string token;
  while (std::getline(ss, token, ',')) {
    int num = std::stoi(token); // Convert the string token to an integer
    int_set.insert(num);        // Insert the integer into the vector
  }
  return int_set;
}

std::map<std::string, std::string> get_configs(std::string path) {
  std::map<std::string, std::string>
      myMap;                // Define a map with string keys and string values
  std::ifstream file(path); // Open the text file for reading
  if (file.is_open()) {
    std::string line;
    while (std::getline(file, line)) {  // Read each line from the file
      std::size_t pos = line.find("="); // Find the position of the equals sign
      if (pos != std::string::npos) {
        std::string key =
            line.substr(0, pos); // Extract the key before the equals sign
        std::string value =
            line.substr(pos + 1); // Extract the value after the equals sign
        removeSpaces(key);        // Remove spaces from the key
        removeSpaces(value);      // Remove spaces from the value
        myMap[key] = value;       // Insert the key-value pair into the map
      }
    }
    file.close(); // Close the file
  } else {
    std::cerr << "Failed to open file." << std::endl;
    return myMap;
  }
  return myMap;
}
