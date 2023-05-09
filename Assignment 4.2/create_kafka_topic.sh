#creates a Kafka topic named "events" with 3 partitions and a replication factor of 1.
docker-compose exec broker kafka-topics --create --topic events --partitions 3 --replication-factor 1 --bootstrap-server broker:9092

#creates another Kafka topic named "test" with 3 partitions and a replication factor of 1.
docker-compose exec broker kafka-topics --create --topic test --partitions 3 --replication-factor 1 --bootstrap-server broker:9092

#lists all the Kafka topics available on the Kafka broker running on the local machine. 
docker-compose exec broker kafka-topics --list --bootstrap-server localhost:9092

#Inspect topics, "events" and "test". to retrieve information such as the number of partitions, replication factor, leader, for each topic.
docker-compose exec broker kafka-topics --describe --topic events,test --bootstrap-server localhost:9092
