public class LoneGroupAssertionGraph extends FlowElementGraph { public LoneGroupAssertionGraph() { Pipe pipe = new Pipe( "test" ); pipe = new Each( pipe, AssertionLevel.STRICT, new AssertNotNull() ); pipe = new Each( pipe, new Fields( "line" ), new RegexFilter( "^POST" ) ); pipe = new Each( pipe, new Fields( "line" ), AssertionLevel.STRICT, new AssertMatches( "^POST" ) ); pipe = new GroupBy( pipe, new Fields( "line" ) ); pipe = new Every( pipe, AssertionLevel.STRICT, new AssertGroupSizeEquals( 7L ) ); Map<String, Tap> sources = createHashMap(); sources.put( "test", new NonTap( new Fields( "line" ) ) ); Map<String, Tap> sinks = createHashMap(); sinks.put( "test", new NonTap() ); initialize( sources, sinks, pipe ); } }