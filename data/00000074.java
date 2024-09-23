public class ConnectionFactoryBuilder { protected OperationQueueFactory opQueueFactory ; protected OperationQueueFactory readQueueFactory ; protected OperationQueueFactory writeQueueFactory ; protected Transcoder < Object > transcoder ; protected FailureMode failureMode ; protected Collection < ConnectionObserver > initialObservers = Collections . emptyList ( ) ; protected OperationFactory opFact ; protected Locator locator = Locator . ARRAY_MOD ; protected long opTimeout = -1 ; protected boolean isDaemon = false ; protected boolean shouldOptimize = false ; protected boolean useNagle = false ; protected boolean keepAlive = false ; protected long maxReconnectDelay = DefaultConnectionFactory . DEFAULT_MAX_RECONNECT_DELAY ; protected int readBufSize = -1 ; protected HashAlgorithm hashAlg ; protected AuthDescriptor authDescriptor = null ; protected long opQueueMaxBlockTime = -1 ; protected int timeoutExceptionThreshold = DefaultConnectionFactory . DEFAULT_MAX_TIMEOUTEXCEPTION_THRESHOLD ; protected MetricType metricType = null ; protected MetricCollector collector = null ; protected ExecutorService executorService = null ; protected long authWaitTime = DefaultConnectionFactory . DEFAULT_AUTH_WAIT_TIME ; public ConnectionFactoryBuilder ( ) { } public ConnectionFactoryBuilder ( ConnectionFactory cf ) { setAuthDescriptor ( cf . getAuthDescriptor ( ) ) ; setDaemon ( cf . isDaemon ( ) ) ; setFailureMode ( cf . getFailureMode ( ) ) ; setHashAlg ( cf . getHashAlg ( ) ) ; setInitialObservers ( cf . getInitialObservers ( ) ) ; setMaxReconnectDelay ( cf . getMaxReconnectDelay ( ) ) ; setOpQueueMaxBlockTime ( cf . getOpQueueMaxBlockTime ( ) ) ; setOpTimeout ( cf . getOperationTimeout ( ) ) ; setReadBufferSize ( cf . getReadBufSize ( ) ) ; setShouldOptimize ( cf . shouldOptimize ( ) ) ; setTimeoutExceptionThreshold ( cf . getTimeoutExceptionThreshold ( ) ) ; setTranscoder ( cf . getDefaultTranscoder ( ) ) ; setUseNagleAlgorithm ( cf . useNagleAlgorithm ( ) ) ; setEnableMetrics ( cf . enableMetrics ( ) ) ; setListenerExecutorService ( cf . getListenerExecutorService ( ) ) ; setAuthWaitTime ( cf . getAuthWaitTime ( ) ) ; } public ConnectionFactoryBuilder setOpQueueFactory ( OperationQueueFactory q ) { opQueueFactory = q ; return this ; } public ConnectionFactoryBuilder setReadOpQueueFactory ( OperationQueueFactory q ) { readQueueFactory = q ; return this ; } public ConnectionFactoryBuilder setWriteOpQueueFactory ( OperationQueueFactory q ) { writeQueueFactory = q ; return this ; } public ConnectionFactoryBuilder setOpQueueMaxBlockTime ( long t ) { opQueueMaxBlockTime = t ; return this ; } public ConnectionFactoryBuilder setTranscoder ( Transcoder < Object > t ) { transcoder = t ; return this ; } public ConnectionFactoryBuilder setFailureMode ( FailureMode fm ) { failureMode = fm ; return this ; } public ConnectionFactoryBuilder setInitialObservers ( Collection < ConnectionObserver > obs ) { initialObservers = obs ; return this ; } public ConnectionFactoryBuilder setOpFact ( OperationFactory f ) { opFact = f ; return this ; } public ConnectionFactoryBuilder setOpTimeout ( long t ) { opTimeout = t ; return this ; } public ConnectionFactoryBuilder setDaemon ( boolean d ) { isDaemon = d ; return this ; } public ConnectionFactoryBuilder setShouldOptimize ( boolean o ) { shouldOptimize = o ; return this ; } public ConnectionFactoryBuilder setReadBufferSize ( int to ) { readBufSize = to ; return this ; } public ConnectionFactoryBuilder setHashAlg ( HashAlgorithm to ) { hashAlg = to ; return this ; } public ConnectionFactoryBuilder setUseNagleAlgorithm ( boolean to ) { useNagle = to ; return this ; } public ConnectionFactoryBuilder setKeepAlive ( boolean on ) { keepAlive = on ; return this ; } public ConnectionFactoryBuilder setProtocol ( Protocol prot ) { switch ( prot ) { case TEXT : opFact = new AsciiOperationFactory ( ) ; break ; case BINARY : opFact = new BinaryOperationFactory ( ) ; break ; default : assert false : "Unhandled protocol : " + prot ; } return this ; } public ConnectionFactoryBuilder setLocatorType ( Locator l ) { locator = l ; return this ; } public ConnectionFactoryBuilder setMaxReconnectDelay ( long to ) { assert to > 0 : "Reconnect delay must be a positive number" ; maxReconnectDelay = to ; return this ; } public ConnectionFactoryBuilder setAuthDescriptor ( AuthDescriptor to ) { authDescriptor = to ; return this ; } public ConnectionFactoryBuilder setTimeoutExceptionThreshold ( int to ) { assert to > 1 : "Minimum timeout exception threshold is 2" ; if ( to > 1 ) { timeoutExceptionThreshold = to - 2 ; } return this ; } public ConnectionFactoryBuilder setEnableMetrics ( MetricType type ) { metricType = type ; return this ; } public ConnectionFactoryBuilder setMetricCollector ( MetricCollector collector ) { this . collector = collector ; return this ; } public ConnectionFactoryBuilder setListenerExecutorService ( ExecutorService executorService ) { this . executorService = executorService ; return this ; } public ConnectionFactoryBuilder setAuthWaitTime ( long authWaitTime ) { this . authWaitTime = authWaitTime ; return this ; } public ConnectionFactory build ( ) { return new DefaultConnectionFactory ( ) { @ Override public BlockingQueue < Operation > createOperationQueue ( ) { return opQueueFactory == null ? super . createOperationQueue ( ) : opQueueFactory . create ( ) ; } @ Override public BlockingQueue < Operation > createReadOperationQueue ( ) { return readQueueFactory == null ? super . createReadOperationQueue ( ) : readQueueFactory . create ( ) ; } @ Override public BlockingQueue < Operation > createWriteOperationQueue ( ) { return writeQueueFactory == null ? super . createReadOperationQueue ( ) : writeQueueFactory . create ( ) ; } @ Override public NodeLocator createLocator ( List < MemcachedNode > nodes ) { switch ( locator ) { case ARRAY_MOD : return new ArrayModNodeLocator ( nodes , getHashAlg ( ) ) ; case CONSISTENT : return new KetamaNodeLocator ( nodes , getHashAlg ( ) ) ; default : throw new IllegalStateException ( "Unhandled locator type : " + locator ) ; } } @ Override public Transcoder < Object > getDefaultTranscoder ( ) { return transcoder == null ? super . getDefaultTranscoder ( ) : transcoder ; } @ Override public FailureMode getFailureMode ( ) { return failureMode == null ? super . getFailureMode ( ) : failureMode ; } @ Override public HashAlgorithm getHashAlg ( ) { return hashAlg == null ? super . getHashAlg ( ) : hashAlg ; } public Collection < ConnectionObserver > getInitialObservers ( ) { return initialObservers ; } @ Override public OperationFactory getOperationFactory ( ) { return opFact == null ? super . getOperationFactory ( ) : opFact ; } @ Override public long getOperationTimeout ( ) { return opTimeout == -1 ? super . getOperationTimeout ( ) : opTimeout ; } @ Override public int getReadBufSize ( ) { return readBufSize == -1 ? super . getReadBufSize ( ) : readBufSize ; } @ Override public boolean isDaemon ( ) { return isDaemon ; } @ Override public boolean shouldOptimize ( ) { return shouldOptimize ; } public boolean getKeepAlive ( ) { return keepAlive ; } @ Override public boolean useNagleAlgorithm ( ) { return useNagle ; } @ Override public long getMaxReconnectDelay ( ) { return maxReconnectDelay ; } @ Override public AuthDescriptor getAuthDescriptor ( ) { return authDescriptor ; } @ Override public long getOpQueueMaxBlockTime ( ) { return opQueueMaxBlockTime > -1 ? opQueueMaxBlockTime : super . getOpQueueMaxBlockTime ( ) ; } @ Override public int getTimeoutExceptionThreshold ( ) { return timeoutExceptionThreshold ; } @ Override public MetricType enableMetrics ( ) { return metricType == null ? super . enableMetrics ( ) : metricType ; } @ Override public MetricCollector getMetricCollector ( ) { return collector == null ? super . getMetricCollector ( ) : collector ; } @ Override public ExecutorService getListenerExecutorService ( ) { return executorService == null ? super . getListenerExecutorService ( ) : executorService ; } @ Override public boolean isDefaultExecutorService ( ) { return executorService == null ; } @ Override public long getAuthWaitTime ( ) { return authWaitTime ; } } ; } public static enum Protocol { TEXT , BINARY } public static enum Locator { ARRAY_MOD , CONSISTENT , VBUCKET } }