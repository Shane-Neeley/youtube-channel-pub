// MongoDB queries into the bears dataset

//////////////////
// find -- know your enemies
db.bears.find({})

//////////////////
// say you take a vacation every July, and only after 8pm, so you only care about bears then.
var q = {
  "observed_on_details.month": 7,
  "observed_on_details.hour": {$gte: 20}
}
db.bears.find(q)
// just one bear you gotta worry about, lets take a look at it
db.bears.find(q).forEach(function(b) {
    b.photos.forEach(function(p) {
        print(p.url)
    })
})

//////////////////
// distinct - get the unique items
// lets look at all photos of bears
db.bears.distinct("photos.url", {})
// "https://static.inaturalist.org/photos/46749280/square.jpeg?1564604461",
// "https://static.inaturalist.org/photos/46598933/square.jpg?1564483523",
// "https://static.inaturalist.org/photos/46269172/square.jpeg?1564241134"
// ...

//////////////////
// count, how many you're dealing with aroung Glacier National Park
db.bears.count({place_guess: {$regex: /glacier national/i}})
// 2!!!

//////////////////
// forEach bear, run the hell away

//////////////////
// print \t and copy to google spreadsheet - takes docs into human format.
// some humans don't like JSON, they like spreadsheets. To keep these humans safe,
// print tab seperated data and copy it over for them.
var headers = ['Lat', 'Lon', 'Month', 'Year', 'Identifications']
print(headers.join('\t'))
db.bears.find({}).forEach(function(b) {
  var total = []
  if (b.geojson) {
    total.push(b.geojson.coordinates[0])
    total.push(b.geojson.coordinates[1])
  } else {
    total.push('idk')
    total.push('idk')
  }
  total.push(b.observed_on_details.month)
  total.push(b.observed_on_details.year)
  var ids = []
  b.identifications.forEach(function(id) {
    // WHO CARES about extinct BEARS right?
    if (!id.taxon.extinct) {
        ids.push(id.taxon.name)
    }
  })
  total.push(ids.join(';'))
  print(total.join('\t'))
})
// copy to spreadsheet
// Lat	Lon	Month	Year	Identifications
// 7.4477861635	46.9520126079	7	2019	Ursus
// -103.2651589	20.6216356	7	2019	Ursus
// -113.4243644761	48.6510135924	7	2019	Ursus arctos horribilis;Ursus americanus
// ...

/////////////////
// put an index on the taxon name so it queries faster
db.bears.createIndex({"identifications.taxon.name": 1})
// put a unique index on the observation id so it queries helllllaaa fast
db.bears.createIndex({"id": 1}, {unique: true})

//////////////////
// update / set / multi
// set a flag on the americanus bears to say that they're not all that bad
// give the field an awful naming convention :-(
db.bears.update(
  {"identifications.taxon.name": "Ursus americanus"},
  {$set: {"notThat_bad-of-Guys": true}},
  {multi: true}
)
db.bears.count({"notThat_bad-of-Guys": {$ne: null}})

//////////////////
// regex queries w/ escape .. escape stuff so your regex don't fail.
var escaped = "HAMBON'E*(ss...)".replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
db.bears.find({somethingSometime: new RegExp("^" + escaped + "$", 'i')})
