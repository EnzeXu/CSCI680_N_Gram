public class BinaryCancellationTest extends CancellationBaseCase { @ Override protected void initClient ( ) throws Exception { initClient ( new BinaryConnectionFactory ( ) { @ Override public FailureMode getFailureMode ( ) { return FailureMode . Retry ; } } ) ; } }