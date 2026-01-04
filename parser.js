const fs = require('fs');

class Parser {
  constructor(filePath) {
    this.filePath = filePath;
  }

  readData() {
    try {
      const data = fs.readFileSync(this.filePath, 'utf8');
      return JSON.parse(data);
    } catch (err) {
      throw new Error(`Error reading file: ${err}`);
    }
  }

  parseData(data) {
    if (!Array.isArray(data)) {
      throw new Error('Invalid data format');
    }

    const parsedData = data.map((record) => {
      if (typeof record !== 'object') {
        throw new Error('Invalid data format');
      }

      return record;
    });

    return parsedData;
  }
}

module.exports = Parser;