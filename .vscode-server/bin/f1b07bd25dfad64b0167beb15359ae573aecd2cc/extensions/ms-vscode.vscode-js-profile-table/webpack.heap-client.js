module.exports = {
  ...require('../../scripts/webpack.client')(__dirname, 'heap-client'),
  entry: `./src/heap-client/client.tsx`,
};
