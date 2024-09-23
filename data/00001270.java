public class JoinAroundJoinRightMostGraphSwapped extends FlowElementGraph { public JoinAroundJoinRightMostGraphSwapped() { Function function = new Insert( new Fields( "num", "char" ), "a", "b" ); Pipe pipeLower = new Each( new Pipe( "lower" ), new Fields( "line" ), function ); Pipe pipeUpper1 = new Each( new Pipe( "upper1" ), new Fields( "line" ), function ); Pipe pipeUpper2 = new Each( new Pipe( "upper2" ), new Fields( "line" ), function ); Pipe splice1 = new HashJoin( pipeLower, new Fields( "num" ), pipeUpper1, new Fields( "num" ), new Fields( "num1", "char1", "num2", "char2" ) ); splice1 = new Each( splice1, new Identity() ); Pipe splice2 = new HashJoin( pipeUpper2, new Fields( "num1" ), splice1, new Fields( "num" ), new Fields( "num1", "char1", "num2", "char2", "num3", "char3" ) ); Map<String, Tap> sources = createHashMap(); sources.put( "lower", new NonTap( new Fields( "offset", "line" ) ) ); NonTap shared = new NonTap( new Fields( "offset", "line" ) ); sources.put( "upper1", shared ); sources.put( "upper2", shared ); Map<String, Tap> sinks = createHashMap(); sinks.put( splice2.getName(), new NonTap( new Fields( "offset", "line" ) ) ); initialize( sources, sinks, splice2 ); } }