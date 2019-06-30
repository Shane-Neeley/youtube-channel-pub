var request = require('request')
var async = require('async')

async.series([
  function (cbk) {
    console.log('im just a thing')
    cbk()
  },
  function (cbk) {
    request('www.ham.com', function (err, response) {
      if (err) return cbk(err)
      if (response.statusCode == 200) {
        console.log('shiiiba innuusss')
      }
      cbk()
    })
  },
  function (cbk) {
    cbk()
  }
], function (err) {
  if (err) throw err
  // whoa im at the end of it all
  process.exit()
})
