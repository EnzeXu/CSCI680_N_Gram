public class HadoopAdapterTapPlatformTest extends PlatformTestCase { public HadoopAdapterTapPlatformTest ( ) { super ( true , 5 , 3 ) ; } @ Test public void testWriteReadHDFS ( ) throws Exception { copyFromLocal ( inputFileApache ) ; Tap source = new FileTap ( new cascading . scheme . local . TextLine ( new Fields ( "offset" , "line" ) ) , inputFileApache ) ; Tap intermediate = new LocalHfsAdaptor ( new Hfs ( new cascading . scheme . hadoop . TextLine ( ) , getOutputPath ( "/intermediate" ) , SinkMode . REPLACE ) ) ; Tap sink = new FileTap ( new cascading . scheme . local . TextLine ( ) , getOutputPath ( "/final" ) , SinkMode . REPLACE ) ; Pipe pipe = new Pipe ( "test" ) ; Flow first = new LocalFlowConnector ( getPlatform ( ) . getProperties ( ) ) . connect ( source , intermediate , pipe ) ; first . complete ( ) ; validateLength ( first , 10 ) ; Flow second = new LocalFlowConnector ( getPlatform ( ) . getProperties ( ) ) . connect ( intermediate , sink , pipe ) ; second . complete ( ) ; validateLength ( second , 10 ) ; } @ Test public void testWriteReadHDFSMultiSource ( ) throws Exception { copyFromLocal ( inputFileApache ) ; Tap source = new MultiSourceTap ( new FileTap ( new cascading . scheme . local . TextLine ( new Fields ( "offset" , "line" ) ) , inputFileApache ) , new FileTap ( new cascading . scheme . local . TextLine ( new Fields ( "offset" , "line" ) ) , inputFileApache ) ) ; Tap intermediate = new LocalHfsAdaptor ( new Hfs ( new cascading . scheme . hadoop . TextLine ( ) , getOutputPath ( "/intermediate" ) , SinkMode . REPLACE ) ) ; Tap sink = new FileTap ( new cascading . scheme . local . TextLine ( ) , getOutputPath ( "/final" ) , SinkMode . REPLACE ) ; Pipe pipe = new Pipe ( "test" ) ; Flow first = new LocalFlowConnector ( getPlatform ( ) . getProperties ( ) ) . connect ( source , intermediate , pipe ) ; first . complete ( ) ; validateLength ( first , 20 ) ; Flow second = new LocalFlowConnector ( getPlatform ( ) . getProperties ( ) ) . connect ( intermediate , sink , pipe ) ; second . complete ( ) ; validateLength ( second , 20 ) ; } @ Test public void testPartitionedWriteReadHDFS ( ) throws Exception { copyFromLocal ( inputFileLhs ) ; Tap source = new FileTap ( new cascading . scheme . local . TextDelimited ( new Fields ( "num" , "char" ) , " " ) , inputFileLhs ) ; Hfs original = new Hfs ( new TextDelimited ( new Fields ( "num" , "char" ) , " " ) , getOutputPath ( "/intermediate" ) , SinkMode . REPLACE ) ; Tap intermediate = new LocalHfsAdaptor ( new PartitionTap ( original , new DelimitedPartition ( new Fields ( "num" ) , "/" ) ) ) ; Tap sink = new FileTap ( new cascading . scheme . local . TextDelimited ( new Fields ( "num" , "char" ) , " " ) , getOutputPath ( "/final" ) , SinkMode . REPLACE ) ; Pipe pipe = new Pipe ( "test" ) ; Flow first = new LocalFlowConnector ( getPlatform ( ) . getProperties ( ) ) . connect ( source , intermediate , pipe ) ; first . complete ( ) ; validateLength ( first , 13 ) ; Flow second = new LocalFlowConnector ( getPlatform ( ) . getProperties ( ) ) . connect ( intermediate , sink , pipe ) ; second . complete ( ) ; validateLength ( second , 13 ) ; } }