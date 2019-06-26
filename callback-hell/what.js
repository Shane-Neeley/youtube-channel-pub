var request = require('request')
var async = require('async')

async.series([
  function (cbk) {
    cbk()
  },
  function (cbk) {
    cbk()
  }
], function (err) {
  if (err) throw err
  process.exit()
})
