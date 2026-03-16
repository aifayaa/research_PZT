const os = require('os');

console.log('CPU Information:');
os.cpus().forEach((cpu, index) => {
  console.log(`CPU ${index}: ${cpu.model} - ${cpu.speed} MHz`);
});

console.log(`\nTotal Memory: ${(os.totalmem() / (1024 * 1024 * 1024)).toFixed(2)} GB`);
console.log(`Free Memory: ${(os.freemem() / (1024 * 1024 * 1024)).toFixed(2)} GB`);

// Optional: Check platform info
console.log(`\nPlatform: ${os.platform()} ${os.arch()}`);
