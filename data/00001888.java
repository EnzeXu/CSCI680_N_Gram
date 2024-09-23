public class KafkaTap<K, V> extends Tap<Properties, KafkaConsumerRecordIterator<K, V>, Producer<K, V>> { private static final Logger LOG = LoggerFactory.getLogger( KafkaTap.class ); public static final Properties CONSUME_AUTO_COMMIT_LATEST = new Properties() { { setProperty( ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "true" ); setProperty( ConsumerConfig.AUTO_COMMIT_INTERVAL_MS_CONFIG, "1000" ); setProperty( ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "latest" ); } }; public static final Properties CONSUME_AUTO_COMMIT_EARLIEST = new Properties() { { setProperty( ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "true" ); setProperty( ConsumerConfig.AUTO_COMMIT_INTERVAL_MS_CONFIG, "1000" ); setProperty( ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest" ); } }; public static final Properties PRODUCE_ACK_ALL_NO_RETRY = new Properties() { { setProperty( ProducerConfig.ACKS_CONFIG, "all" ); setProperty( ProducerConfig.RETRIES_CONFIG, "0" ); } }; public static final long DEFAULT_POLL_TIMEOUT = 10_000L; public static final short DEFAULT_REPLICATION_FACTOR = 1; public static final int DEFAULT_NUM_PARTITIONS = 1; Properties defaultProperties = PropertyUtil.merge( CONSUME_AUTO_COMMIT_EARLIEST, PRODUCE_ACK_ALL_NO_RETRY ); String hostname; String[] topics; boolean isTopicPattern = false; int numPartitions = DEFAULT_NUM_PARTITIONS; short replicationFactor = DEFAULT_REPLICATION_FACTOR; String clientID = null; String groupID = Tap.id( this ); long pollTimeout = DEFAULT_POLL_TIMEOUT; public static URI makeURI( String hostname, String... topics ) { if( hostname == null ) throw new IllegalArgumentException( "hostname may not be null" ); Arrays.sort( topics ); try { return new URI( "kafka", hostname, "/" + Util.join( ",", topics ), null, null ); } catch( URISyntaxException exception ) { throw new IllegalArgumentException( exception.getMessage(), exception ); } } public KafkaTap( Properties defaultProperties, KafkaScheme<K, V, ?, ?> scheme, URI identifier ) { this( defaultProperties, scheme, identifier, DEFAULT_POLL_TIMEOUT, DEFAULT_NUM_PARTITIONS, DEFAULT_REPLICATION_FACTOR ); } public KafkaTap( KafkaScheme<K, V, ?, ?> scheme, URI identifier, long pollTimeout ) { this( scheme, identifier, pollTimeout, DEFAULT_NUM_PARTITIONS, DEFAULT_REPLICATION_FACTOR ); } public KafkaTap( KafkaScheme<K, V, ?, ?> scheme, URI identifier, int numPartitions, short replicationFactor ) { this( scheme, identifier, DEFAULT_POLL_TIMEOUT, numPartitions, replicationFactor ); } public KafkaTap( KafkaScheme<K, V, ?, ?> scheme, URI identifier, long pollTimeout, int numPartitions, short replicationFactor ) { this( null, scheme, identifier, pollTimeout, numPartitions, replicationFactor ); } public KafkaTap( KafkaScheme<K, V, ?, ?> scheme, URI identifier ) { this( scheme, identifier, DEFAULT_POLL_TIMEOUT, DEFAULT_NUM_PARTITIONS, DEFAULT_REPLICATION_FACTOR ); } public KafkaTap( Properties defaultProperties, KafkaScheme<K, V, ?, ?> scheme, URI identifier, long pollTimeout ) { this( defaultProperties, scheme, identifier, pollTimeout, DEFAULT_NUM_PARTITIONS, DEFAULT_REPLICATION_FACTOR ); } public KafkaTap( Properties defaultProperties, KafkaScheme<K, V, ?, ?> scheme, URI identifier, int numPartitions, short replicationFactor ) { this( defaultProperties, scheme, identifier, DEFAULT_POLL_TIMEOUT, numPartitions, replicationFactor ); } public KafkaTap( Properties defaultProperties, KafkaScheme<K, V, ?, ?> scheme, URI identifier, long pollTimeout, int numPartitions, short replicationFactor ) { this( defaultProperties, scheme, identifier, null, pollTimeout, numPartitions, replicationFactor ); } public KafkaTap( Properties defaultProperties, KafkaScheme<K, V, ?, ?> scheme, URI identifier, String clientID ) { this( defaultProperties, scheme, identifier, clientID, DEFAULT_POLL_TIMEOUT, DEFAULT_NUM_PARTITIONS, DEFAULT_REPLICATION_FACTOR ); } public KafkaTap( Properties defaultProperties, KafkaScheme<K, V, ?, ?> scheme, URI identifier, String clientID, String groupID ) { this( defaultProperties, scheme, identifier, clientID, groupID, DEFAULT_POLL_TIMEOUT, DEFAULT_NUM_PARTITIONS, DEFAULT_REPLICATION_FACTOR ); } public KafkaTap( KafkaScheme<K, V, ?, ?> scheme, URI identifier, String clientID, long pollTimeout ) { this( scheme, identifier, clientID, pollTimeout, DEFAULT_NUM_PARTITIONS, DEFAULT_REPLICATION_FACTOR ); } public KafkaTap( KafkaScheme<K, V, ?, ?> scheme, URI identifier, String clientID, int numPartitions, short replicationFactor ) { this( scheme, identifier, clientID, DEFAULT_POLL_TIMEOUT, numPartitions, replicationFactor ); } public KafkaTap( KafkaScheme<K, V, ?, ?> scheme, URI identifier, String clientID, long pollTimeout, int numPartitions, short replicationFactor ) { this( null, scheme, identifier, clientID, pollTimeout, numPartitions, replicationFactor ); } public KafkaTap( KafkaScheme<K, V, ?, ?> scheme, URI identifier, String clientID ) { this( scheme, identifier, clientID, DEFAULT_POLL_TIMEOUT, DEFAULT_NUM_PARTITIONS, DEFAULT_REPLICATION_FACTOR ); } public KafkaTap( KafkaScheme<K, V, ?, ?> scheme, URI identifier, String clientID, String groupID ) { this( null, scheme, identifier, clientID, groupID, DEFAULT_POLL_TIMEOUT, DEFAULT_NUM_PARTITIONS, DEFAULT_REPLICATION_FACTOR ); } public KafkaTap( Properties defaultProperties, KafkaScheme<K, V, ?, ?> scheme, URI identifier, String clientID, long pollTimeout ) { this( defaultProperties, scheme, identifier, clientID, pollTimeout, DEFAULT_NUM_PARTITIONS, DEFAULT_REPLICATION_FACTOR ); } public KafkaTap( Properties defaultProperties, KafkaScheme<K, V, ?, ?> scheme, URI identifier, String clientID, int numPartitions, short replicationFactor ) { this( defaultProperties, scheme, identifier, clientID, DEFAULT_POLL_TIMEOUT, numPartitions, replicationFactor ); } public KafkaTap( Properties defaultProperties, KafkaScheme<K, V, ?, ?> scheme, URI identifier, String clientID, long pollTimeout, int numPartitions, short replicationFactor ) { this( defaultProperties, scheme, identifier, clientID, null, pollTimeout, numPartitions, replicationFactor ); } public KafkaTap( Properties defaultProperties, KafkaScheme<K, V, ?, ?> scheme, URI identifier, String clientID, String groupID, long pollTimeout, int numPartitions, short replicationFactor ) { super( scheme, SinkMode.UPDATE ); if( defaultProperties != null ) this.defaultProperties = new Properties( defaultProperties ); if( identifier == null ) throw new IllegalArgumentException( "identifier may not be null" ); if( !identifier.getScheme().equalsIgnoreCase( "kafka" ) ) throw new IllegalArgumentException( "identifier does not have kafka scheme" ); this.hostname = identifier.getHost(); if( identifier.getPort() != -1 ) this.hostname += ":" + identifier.getPort(); if( identifier.getQuery() == null ) throw new IllegalArgumentException( "must have at least one topic in the query part of the URI" ); if( clientID != null ) this.clientID = clientID; if( groupID != null ) this.groupID = groupID; this.pollTimeout = pollTimeout; this.numPartitions = numPartitions; this.replicationFactor = replicationFactor; applyTopics( identifier.getQuery().split( "," ) ); } public KafkaTap( KafkaScheme<K, V, ?, ?> scheme, String hostname, long pollTimeout, String... topics ) { this( scheme, hostname, pollTimeout, DEFAULT_NUM_PARTITIONS, DEFAULT_REPLICATION_FACTOR, topics ); } public KafkaTap( KafkaScheme<K, V, ?, ?> scheme, String hostname, long pollTimeout, int numPartitions, short replicationFactor, String... topics ) { this( null, scheme, hostname, pollTimeout, numPartitions, replicationFactor, topics ); } public KafkaTap( Properties defaultProperties, KafkaScheme<K, V, ?, ?> scheme, String hostname, int numPartitions, short replicationFactor, String... topics ) { this( defaultProperties, scheme, hostname, DEFAULT_POLL_TIMEOUT, numPartitions, replicationFactor, topics ); } public KafkaTap( Properties defaultProperties, KafkaScheme<K, V, ?, ?> scheme, String hostname, String... topics ) { this( defaultProperties, scheme, hostname, DEFAULT_POLL_TIMEOUT, DEFAULT_NUM_PARTITIONS, DEFAULT_REPLICATION_FACTOR, topics ); } public KafkaTap( Properties defaultProperties, KafkaScheme<K, V, ?, ?> scheme, String hostname, long pollTimeout, String... topics ) { this( defaultProperties, scheme, hostname, pollTimeout, DEFAULT_NUM_PARTITIONS, DEFAULT_REPLICATION_FACTOR, topics ); } public KafkaTap( Properties defaultProperties, KafkaScheme<K, V, ?, ?> scheme, String hostname, long pollTimeout, int numPartitions, short replicationFactor, String... topics ) { this( defaultProperties, scheme, hostname, null, pollTimeout, numPartitions, replicationFactor, topics ); } public KafkaTap( KafkaScheme<K, V, ?, ?> scheme, String hostname, String clientID, String... topics ) { this( scheme, hostname, clientID, DEFAULT_POLL_TIMEOUT, DEFAULT_NUM_PARTITIONS, DEFAULT_REPLICATION_FACTOR, topics ); } public KafkaTap( KafkaScheme<K, V, ?, ?> scheme, String hostname, String clientID, long pollTimeout, String... topics ) { this( scheme, hostname, clientID, pollTimeout, DEFAULT_NUM_PARTITIONS, DEFAULT_REPLICATION_FACTOR, topics ); } public KafkaTap( KafkaScheme<K, V, ?, ?> scheme, String hostname, String clientID, long pollTimeout, int numPartitions, short replicationFactor, String... topics ) { this( null, scheme, hostname, clientID, pollTimeout, numPartitions, replicationFactor, topics ); } public KafkaTap( Properties defaultProperties, KafkaScheme<K, V, ?, ?> scheme, String hostname, String clientID, int numPartitions, short replicationFactor, String... topics ) { this( defaultProperties, scheme, hostname, clientID, DEFAULT_POLL_TIMEOUT, numPartitions, replicationFactor, topics ); } public KafkaTap( Properties defaultProperties, KafkaScheme<K, V, ?, ?> scheme, String hostname, String clientID, String... topics ) { this( defaultProperties, scheme, hostname, clientID, DEFAULT_POLL_TIMEOUT, DEFAULT_NUM_PARTITIONS, DEFAULT_REPLICATION_FACTOR, topics ); } public KafkaTap( Properties defaultProperties, KafkaScheme<K, V, ?, ?> scheme, String hostname, String clientID, long pollTimeout, String... topics ) { this( defaultProperties, scheme, hostname, clientID, pollTimeout, DEFAULT_NUM_PARTITIONS, DEFAULT_REPLICATION_FACTOR, topics ); } public KafkaTap( Properties defaultProperties, KafkaScheme<K, V, ?, ?> scheme, String hostname, String clientID, long pollTimeout, int numPartitions, short replicationFactor, String... topics ) { this( defaultProperties, scheme, hostname, clientID, null, pollTimeout, numPartitions, replicationFactor, topics ); } public KafkaTap( Properties defaultProperties, KafkaScheme<K, V, ?, ?> scheme, String hostname, String clientID, String groupID, long pollTimeout, int numPartitions, short replicationFactor, String... topics ) { super( scheme, SinkMode.UPDATE ); if( defaultProperties != null ) this.defaultProperties = new Properties( defaultProperties ); this.hostname = hostname; if( clientID != null ) this.clientID = clientID; if( groupID != null ) this.groupID = groupID; this.pollTimeout = pollTimeout; this.numPartitions = numPartitions; this.replicationFactor = replicationFactor; applyTopics( topics ); } protected void applyTopics( String[] topics ) { if( topics[ 0 ].matches( "^/([^/]| { this.topics = new String[]{topics[ 0 ].substring( 1, topics[ 0 ].length() - 1 )}; this.isTopicPattern = true; } else { this.topics = new String[ topics.length ]; System.arraycopy( topics, 0, this.topics, 0, topics.length ); } } public String getHostname() { return hostname; } public String getClientID() { return clientID; } public String getGroupID() { return groupID; } public String[] getTopics() { return topics; } public boolean isTopicPattern() { return isTopicPattern; } @Override public String getIdentifier() { return makeURI( hostname, topics ).toString(); } protected Consumer<K, V> createKafkaConsumer( Properties properties ) { return new ForwardingConsumer<>( properties ); } @Override public TupleEntryIterator openForRead( FlowProcess<? extends Properties> flowProcess, KafkaConsumerRecordIterator<K, V> consumerRecord ) throws IOException { Properties props = PropertyUtil.merge( flowProcess.getConfig(), defaultProperties ); props.setProperty( ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, hostname ); Set<String> keys = props.stringPropertyNames(); if( clientID != null && !keys.contains( ConsumerConfig.CLIENT_ID_CONFIG ) ) props.setProperty( ConsumerConfig.CLIENT_ID_CONFIG, clientID ); if( !keys.contains( ConsumerConfig.GROUP_ID_CONFIG ) ) props.setProperty( ConsumerConfig.GROUP_ID_CONFIG, groupID ); sourceConfInit( flowProcess, props ); Properties consumerProperties = PropertyUtil.retain( props, ConsumerConfig.configNames() ); Consumer<K, V> consumer = createKafkaConsumer( consumerProperties ); preConsumerSubscribe( consumer ); if( isTopicPattern ) consumer.subscribe( Pattern.compile( topics[ 0 ] ), getConsumerRebalanceListener( consumer ) ); else consumer.subscribe( Arrays.asList( getTopics() ), getConsumerRebalanceListener( consumer ) ); postConsumerSubscribe( consumer ); CloseableIterator<Iterator<ConsumerRecord<K, V>>> iterator = new CloseableIterator<Iterator<ConsumerRecord<K, V>>>() { boolean completed = false; ConsumerRecords<K, V> records; @Override public boolean hasNext() { if( records != null ) return true; if( completed ) return false; records = consumer.poll( pollTimeout ); if( LOG.isDebugEnabled() ) LOG.debug( "kafka records polled: {}", records.count() ); if( records.isEmpty() ) { completed = true; records = null; } return records != null; } @Override public Iterator<ConsumerRecord<K, V>> next() { if( !hasNext() ) throw new NoSuchElementException( "no more elements" ); try { CloseableIterator<Iterator<ConsumerRecord<K, V>>> parent = this; return new KafkaConsumerRecordIterator<K, V>() { Iterator<ConsumerRecord<K, V>> delegate = records.iterator(); Supplier<Boolean> hasNext = () -> delegate.hasNext(); @Override protected Consumer<K, V> getConsumer() { return consumer; } @Override public void close() throws IOException { hasNext = () -> false; parent.close(); } @Override public boolean hasNext() { return hasNext.get(); } @Override public ConsumerRecord<K, V> next() { return delegate.next(); } }; } finally { records = null; } } @Override public void close() { try { try { consumer.close(); } catch( IllegalStateException exception ) { LOG.error( "ignoring exception on closing", exception ); } } finally { completed = true; } } }; return new TupleEntrySchemeIterator<Properties, Iterator<ConsumerRecord<K, V>>>( flowProcess, this, getScheme(), iterator ); } protected void preConsumerSubscribe( Consumer<K, V> consumer ) { } protected void postConsumerSubscribe( Consumer<K, V> consumer ) { } protected ConsumerRebalanceListener getConsumerRebalanceListener( Consumer<K, V> consumer ) { return new NoOpConsumerRebalanceListener(); } @Override public TupleEntryCollector openForWrite( FlowProcess<? extends Properties> flowProcess, Producer<K, V> producer ) throws IOException { Properties props = PropertyUtil.merge( flowProcess.getConfig(), defaultProperties ); props.setProperty( ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, hostname ); sinkConfInit( flowProcess, props ); producer = new KafkaProducer<>( PropertyUtil.retain( props, ProducerConfig.configNames() ) ); return new TupleEntrySchemeCollector<Properties, Producer<?, ?>>( flowProcess, this, getScheme(), producer ); } protected AdminClient createAdminClient( Properties conf ) { Properties props = new Properties( conf ); props.setProperty( ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, hostname ); return AdminClient.create( props ); } @Override public boolean createResource( Properties conf ) { AdminClient client = createAdminClient( conf ); List<NewTopic> topics = new ArrayList<>( getTopics().length ); for( String topic : getTopics() ) topics.add( new NewTopic( topic, numPartitions, replicationFactor ) ); CreateTopicsResult result = client.createTopics( topics ); KafkaFuture<Void> all = result.all(); try { all.get(); } catch( InterruptedException | ExecutionException exception ) { LOG.info( "unable to create topics" ); return false; } return true; } @Override public boolean deleteResource( Properties conf ) { AdminClient client = createAdminClient( conf ); DeleteTopicsResult result = client.deleteTopics( Arrays.asList( getTopics() ) ); KafkaFuture<Void> all = result.all(); try { all.get(); } catch( InterruptedException | ExecutionException exception ) { LOG.info( "unable to create topics" ); return false; } return true; } @Override public boolean resourceExists( Properties conf ) { AdminClient client = createAdminClient( conf ); DescribeTopicsResult result = client.describeTopics( Arrays.asList( getTopics() ) ); KafkaFuture<Map<String, TopicDescription>> all = result.all(); try { Map<String, TopicDescription> map = all.get(); return map.size() == getTopics().length; } catch( InterruptedException | ExecutionException exception ) { LOG.info( "unable to create topics" ); return false; } } @Override public long getModifiedTime( Properties conf ) throws IOException { if( resourceExists( conf ) ) return Long.MAX_VALUE; else return 0L; } }