## So you wanna learn MongoDB?

Good. You have the right to.

#### Download and Install

Went through that in the [setting up video](http://www.youtube.com/watch?v=ilm-Dt07mQk&t=4m28s)

#### Databases

Start the `mongod` server running on localhost:9000

```
cd ~/mongo/bin
./mongod --dbpath ../data/db
```

Create a package.json. Add the node modules you need. (can skip because it's checked in)
```
cd ~/youtube-channel/mongodb-the-fun-way/
npm init
npm install request
npm install mongodb
npm install async
```

Install packages, and run the acquire script.
```
npm install
node acquire.js
```

#### Collections

`bears` collection: a collection of bears.

#### Queries

Examples in file: `queries.js`

#### Info

__Taxon is threatened__, coordinates obscured by default: One of the taxa suggested in the identifications, or one of the taxa that contain any of these taxa, is known to be rare and/or threatened, so the location of this observation has been obscured.

If you ever find yourself in this area ... Exact coordinates not shown but can get close: [COORDS OF BAD BEAR](https://www.google.com/maps/place/26%C2%B043'49.0%22N+57%C2%B036'12.7%22E/@27.3996755,57.5002643,908879m/data=!3m1!1e3!4m5!3m4!1s0x0:0x0!8m2!3d26.7302686!4d57.6035281)


#### JSON

__For generations, the Hungarians have made the best JSON viewers__

http://jsonviewer.stack.hu/
