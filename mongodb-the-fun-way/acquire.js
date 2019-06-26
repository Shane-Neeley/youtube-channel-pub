// A far better tutorial than this: https://www.w3schools.com/nodejs/nodejs_mongodb_insert.asp
var request = require('request')
var mongo = require('mongodb')
var MongoClient = require('mongodb').MongoClient
var async = require('async')

var mydb

async.series([
  function (cbk) {
    var url = 'mongodb://localhost:27017/'
    MongoClient.connect(url, function (err, db) {
      if (err) cbk(err)
      var dbo = db.db('dangers')
      mydb = dbo
      dbo.createCollection('bears', cbk)
    })
  },
  function (cbk) {
    var genus = 'Ursus'
    var url1 = 'http://api.inaturalist.org/v1/taxa?q=' + genus + '&rank=genus'
    request(url1, function (error, response, body) {
      // console.error('error:', error) // Print the error if one occurred
      // console.log('statusCode:', response && response.statusCode) // Print the response status code if a response was received
      // console.log('body:', body) // Print the json yeaeaaaaa
      body = JSON.parse(body)
      taxon_id = body.results[0].id
      obs_url = [
        'http://api.inaturalist.org/v1/observations?',
        'rank=genus',
        '&taxon_id=' + taxon_id,
        '&page=1', // u want more bears? don't be greedy.
        '&per_page=30&order=desc&order_by=observed_on'
      ].join('')

      request(obs_url, function (error2, response2, body2) {
        body2 = JSON.parse(body2)
        // for each bear!
        async.eachSeries(body2.results, function (bear, cbk2) {
          console.log('OH NO! STAY AWAY FROM: ' + bear.location)
          // put it up in your mongodb
          mydb.collection('bears').insertOne(bear, cbk2)
        }, cbk)
      })
    })
  }
], function (err) {
  if (err) throw err
  // db.close()
  process.exit()
})
