module.exports = {
  ...require('../../scripts/webpack.client')(__dirname, 'cpu-client'),
  entry: `./src/cpu-client/client.tsx`,
};
