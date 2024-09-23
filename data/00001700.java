public class ForwardingConsumer < K , V > implements Consumer < K , V > { Consumer < K , V > consumer ; public ForwardingConsumer ( Properties properties ) { this . consumer = createKafkaConsumerInstance ( properties ) ; } protected KafkaConsumer < K , V > createKafkaConsumerInstance ( Properties properties ) { return new KafkaConsumer < > ( properties ) ; } public Consumer < K , V > getConsumer ( ) { return consumer ; } @ Override public Set < TopicPartition > assignment ( ) { return consumer . assignment ( ) ; } @ Override public Set < String > subscription ( ) { return consumer . subscription ( ) ; } @ Override public void subscribe ( Collection < String > collection ) { consumer . subscribe ( collection ) ; } @ Override public void subscribe ( Collection < String > collection , ConsumerRebalanceListener consumerRebalanceListener ) { consumer . subscribe ( collection , consumerRebalanceListener ) ; } @ Override public void assign ( Collection < TopicPartition > collection ) { consumer . assign ( collection ) ; } @ Override public void subscribe ( Pattern pattern , ConsumerRebalanceListener consumerRebalanceListener ) { consumer . subscribe ( pattern , consumerRebalanceListener ) ; } @ Override public void subscribe ( Pattern pattern ) { consumer . subscribe ( pattern ) ; } @ Override public void unsubscribe ( ) { consumer . unsubscribe ( ) ; } @ Deprecated @ Override public ConsumerRecords < K , V > poll ( long l ) { return consumer . poll ( l ) ; } @ Override public ConsumerRecords < K , V > poll ( Duration duration ) { return consumer . poll ( duration ) ; } @ Override public void commitSync ( ) { consumer . commitSync ( ) ; } @ Override public void commitSync ( Duration duration ) { consumer . commitSync ( duration ) ; } @ Override public void commitSync ( Map < TopicPartition , OffsetAndMetadata > map ) { consumer . commitSync ( map ) ; } @ Override public void commitSync ( Map < TopicPartition , OffsetAndMetadata > map , Duration duration ) { consumer . commitSync ( map , duration ) ; } @ Override public void commitAsync ( ) { consumer . commitAsync ( ) ; } @ Override public void commitAsync ( OffsetCommitCallback offsetCommitCallback ) { consumer . commitAsync ( offsetCommitCallback ) ; } @ Override public void commitAsync ( Map < TopicPartition , OffsetAndMetadata > map , OffsetCommitCallback offsetCommitCallback ) { consumer . commitAsync ( map , offsetCommitCallback ) ; } @ Override public void seek ( TopicPartition topicPartition , long l ) { consumer . seek ( topicPartition , l ) ; } @ Override public void seekToBeginning ( Collection < TopicPartition > collection ) { consumer . seekToBeginning ( collection ) ; } @ Override public void seekToEnd ( Collection < TopicPartition > collection ) { consumer . seekToEnd ( collection ) ; } @ Override public long position ( TopicPartition topicPartition ) { return consumer . position ( topicPartition ) ; } @ Override public long position ( TopicPartition topicPartition , Duration duration ) { return consumer . position ( topicPartition , duration ) ; } @ Override public OffsetAndMetadata committed ( TopicPartition topicPartition ) { return consumer . committed ( topicPartition ) ; } @ Override public OffsetAndMetadata committed ( TopicPartition topicPartition , Duration duration ) { return consumer . committed ( topicPartition , duration ) ; } @ Override public Map < MetricName , ? extends Metric > metrics ( ) { return consumer . metrics ( ) ; } @ Override public List < PartitionInfo > partitionsFor ( String string ) { return consumer . partitionsFor ( string ) ; } @ Override public List < PartitionInfo > partitionsFor ( String string , Duration duration ) { return consumer . partitionsFor ( string , duration ) ; } @ Override public Map < String , List < PartitionInfo > > listTopics ( ) { return consumer . listTopics ( ) ; } @ Override public Map < String , List < PartitionInfo > > listTopics ( Duration duration ) { return consumer . listTopics ( duration ) ; } @ Override public Set < TopicPartition > paused ( ) { return consumer . paused ( ) ; } @ Override public void pause ( Collection < TopicPartition > collection ) { consumer . pause ( collection ) ; } @ Override public void resume ( Collection < TopicPartition > collection ) { consumer . resume ( collection ) ; } @ Override public Map < TopicPartition , OffsetAndTimestamp > offsetsForTimes ( Map < TopicPartition , Long > map ) { return consumer . offsetsForTimes ( map ) ; } @ Override public Map < TopicPartition , OffsetAndTimestamp > offsetsForTimes ( Map < TopicPartition , Long > map , Duration duration ) { return consumer . offsetsForTimes ( map , duration ) ; } @ Override public Map < TopicPartition , Long > beginningOffsets ( Collection < TopicPartition > collection ) { return consumer . beginningOffsets ( collection ) ; } @ Override public Map < TopicPartition , Long > beginningOffsets ( Collection < TopicPartition > collection , Duration duration ) { return consumer . beginningOffsets ( collection , duration ) ; } @ Override public Map < TopicPartition , Long > endOffsets ( Collection < TopicPartition > collection ) { return consumer . endOffsets ( collection ) ; } @ Override public Map < TopicPartition , Long > endOffsets ( Collection < TopicPartition > collection , Duration duration ) { return consumer . endOffsets ( collection , duration ) ; } @ Override public void close ( ) { consumer . close ( ) ; } @ Deprecated @ Override public void close ( long l , TimeUnit timeUnit ) { consumer . close ( l , timeUnit ) ; } @ Override public void close ( Duration duration ) { consumer . close ( duration ) ; } @ Override public void wakeup ( ) { consumer . wakeup ( ) ; } }