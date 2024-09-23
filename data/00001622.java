public class MultiMapReduceFlowPlatformTest extends PlatformTestCase { public MultiMapReduceFlowPlatformTest ( ) { super ( true ) ; } @ Test public void testFlow ( ) throws IOException { getPlatform ( ) . copyFromLocal ( inputFileApache ) ; String outputPath1 = getOutputPath ( "flowTest1" ) ; String outputPath2 = getOutputPath ( "flowTest2" ) ; String outputPath3 = getOutputPath ( "flowTest3" ) ; remove ( outputPath1 , true ) ; remove ( outputPath2 , true ) ; remove ( outputPath3 , true ) ; JobConf defaultConf = ( JobConf ) ( ( BaseHadoopPlatform ) getPlatform ( ) ) . getConfiguration ( ) ; JobConf conf1 = createJob ( defaultConf , "mr1" , InputData . inputFileApache , outputPath1 ) ; JobConf conf2 = createJob ( defaultConf , "mr2" , outputPath1 , outputPath2 ) ; JobConf conf3 = createJob ( defaultConf , "mr3" , outputPath2 , outputPath3 ) ; MultiMapReduceFlow flow = new MultiMapReduceFlow ( "mrflow" , conf1 , conf2 , conf3 ) ; validateLength ( new Hfs ( new TextLine ( ) , InputData . inputFileApache ) . openForRead ( new HadoopFlowProcess ( defaultConf ) ) , 10 ) ; flow . complete ( ) ; validateLength ( new Hfs ( new TextLine ( ) , outputPath1 ) . openForRead ( new HadoopFlowProcess ( defaultConf ) ) , 10 ) ; Collection < Tap > sinks = flow . getSinks ( ) . values ( ) ; assertEquals ( 1 , sinks . size ( ) ) ; String identifier = sinks . iterator ( ) . next ( ) . getIdentifier ( ) ; assertEquals ( "flowTest3" , identifier . substring ( identifier . lastIndexOf ( '/' ) + 1 ) ) ; } @ Test public void testFlowLazy ( ) throws IOException { getPlatform ( ) . copyFromLocal ( inputFileApache ) ; String outputPath1 = getOutputPath ( "flowTest1" ) ; String outputPath2 = getOutputPath ( "flowTest2" ) ; String outputPath3 = getOutputPath ( "flowTest3" ) ; remove ( outputPath1 , true ) ; remove ( outputPath2 , true ) ; remove ( outputPath3 , true ) ; JobConf defaultConf = ( JobConf ) ( ( BaseHadoopPlatform ) getPlatform ( ) ) . getConfiguration ( ) ; JobConf conf1 = createJob ( defaultConf , "mr1" , InputData . inputFileApache , outputPath1 ) ; JobConf conf2 = createJob ( defaultConf , "mr2" , outputPath1 , outputPath2 ) ; JobConf conf3 = createJob ( defaultConf , "mr3" , outputPath2 , outputPath3 ) ; validateLength ( new Hfs ( new TextLine ( ) , InputData . inputFileApache ) . openForRead ( new HadoopFlowProcess ( defaultConf ) ) , 10 ) ; MultiMapReduceFlow flow = new MultiMapReduceFlow ( "mrflow" , conf1 ) ; flow . start ( ) ; Util . safeSleep ( 3000 ) ; flow . attachFlowStep ( conf2 ) ; Util . safeSleep ( 3000 ) ; flow . attachFlowStep ( conf3 ) ; flow . complete ( ) ; validateLength ( new Hfs ( new TextLine ( ) , outputPath1 ) . openForRead ( new HadoopFlowProcess ( defaultConf ) ) , 10 ) ; Collection < Tap > sinks = flow . getSinks ( ) . values ( ) ; assertEquals ( 1 , sinks . size ( ) ) ; String identifier = sinks . iterator ( ) . next ( ) . getIdentifier ( ) ; assertEquals ( "flowTest3" , identifier . substring ( identifier . lastIndexOf ( '/' ) + 1 ) ) ; } @ Test ( expected = IllegalStateException . class ) public void testFlowLazyFail ( ) throws IOException { getPlatform ( ) . copyFromLocal ( inputFileApache ) ; String outputPath1 = getOutputPath ( "flowTest1" ) ; String outputPath2 = getOutputPath ( "flowTest2" ) ; remove ( outputPath1 , true ) ; remove ( outputPath2 , true ) ; JobConf defaultConf = ( JobConf ) ( ( BaseHadoopPlatform ) getPlatform ( ) ) . getConfiguration ( ) ; JobConf conf1 = createJob ( defaultConf , "mr1" , InputData . inputFileApache , outputPath1 ) ; JobConf conf2 = createJob ( defaultConf , "mr2" , outputPath1 , outputPath2 ) ; validateLength ( new Hfs ( new TextLine ( ) , InputData . inputFileApache ) . openForRead ( new HadoopFlowProcess ( defaultConf ) ) , 10 ) ; MultiMapReduceFlow flow = new MultiMapReduceFlow ( "mrflow" , conf1 ) ; flow . complete ( ) ; flow . attachFlowStep ( conf2 ) ; } protected JobConf createJob ( JobConf defaultConf , String name , String inputPath , String outputPath ) { JobConf conf = new JobConf ( defaultConf ) ; conf . setJobName ( name ) ; conf . setOutputKeyClass ( LongWritable . class ) ; conf . setOutputValueClass ( Text . class ) ; conf . setMapperClass ( IdentityMapper . class ) ; conf . setReducerClass ( IdentityReducer . class ) ; conf . setInputFormat ( TextInputFormat . class ) ; conf . setOutputFormat ( TextOutputFormat . class ) ; FileInputFormat . setInputPaths ( conf , new Path ( inputPath ) ) ; FileOutputFormat . setOutputPath ( conf , new Path ( outputPath ) ) ; return conf ; } private String remove ( String path , boolean delete ) throws IOException { FileSystem fs = FileSystem . get ( URI . create ( path ) , HadoopPlanner . createJobConf ( getProperties ( ) ) ) ; if ( delete ) fs . delete ( new Path ( path ) , true ) ; return path ; } }