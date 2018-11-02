use config
db.settings.update({_id: "balancer"}, {$set: {stopped: true}}, true)
while (db.locks.findOne({_id: "balancer", state: {$ne: 0}}) != null) { sleep(1000); }
use config
db.collections.find({_id: /^AirSage\./})
db.collections.remove({_id: /^AirSage\./})
use config
db.databases.find({_id: "AirSage"})
db.databases.remove({_id: "AirSage"})
use config
db.chunks.find({ns: /^AirSage\./})
db.chunks.remove({ns: /^AirSage\./})
use config
var mongoses = db.mongos.find()
while (mongoses.hasNext()) { new Mongo(mongoses.next()._id).getDB("admin").runCommand({flushRouterConfig: 1}) }
use AirSage
db.dropDatabase()
use config
db.settings.update({_id: "balancer"}, {$set: {stopped: false}}, true)
