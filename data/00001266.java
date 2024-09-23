public class HashJoinsIntoMerge extends FlowElementGraph { public HashJoinsIntoMerge() { Map sources = new HashMap(); sources.put( "lower", new NonTap( "lower", new Fields( "offset", "line" ) ) ); sources.put( "upper", new NonTap( "upper", new Fields( "offset", "line" ) ) ); sources.put( "lhs", new NonTap( "lhs", new Fields( "offset", "line" ) ) ); sources.put( "rhs", new NonTap( "rhs", new Fields( "offset", "line" ) ) ); Map sinks = new HashMap(); sinks.put( "sink", new NonTap( "sink", new Fields( "offset", "line" ) ) ); Function splitter = new RegexSplitter( new Fields( "num", "char" ), " " ); Pipe pipeLower = new Each( new Pipe( "lower" ), new Fields( "line" ), splitter ); Pipe pipeUpper = new Each( new Pipe( "upper" ), new Fields( "line" ), splitter ); Pipe pipeLhs = new Each( new Pipe( "lhs" ), new Fields( "line" ), splitter ); Pipe pipeRhs = new Each( new Pipe( "rhs" ), new Fields( "line" ), splitter ); Pipe upperLower = new HashJoin( pipeLower, new Fields( "num" ), pipeUpper, new Fields( "num" ), new Fields( "num1", "char1", "num2", "char2" ) ); upperLower = new Each( upperLower, new Identity() ); Pipe lhsRhs = new HashJoin( pipeLhs, new Fields( "num" ), pipeRhs, new Fields( "num" ), new Fields( "num1", "char1", "num2", "char2" ) ); lhsRhs = new Each( lhsRhs, new Identity() ); Pipe merge = new Merge( "sink", Pipe.pipes( upperLower, lhsRhs ) ); initialize( sources, sinks, merge ); } }