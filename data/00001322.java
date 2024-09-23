public class RegexReplace extends RegexOperation<Pair<Matcher, TupleEntry>> implements Function<Pair<Matcher, TupleEntry>> { private final String replacement; private boolean replaceAll = true; @ConstructorProperties({"fieldDeclaration", "patternString", "replacement", "replaceAll"}) public RegexReplace( Fields fieldDeclaration, String patternString, String replacement, boolean replaceAll ) { this( fieldDeclaration, patternString, replacement ); this.replaceAll = replaceAll; } @ConstructorProperties({"fieldDeclaration", "patternString", "replacement"}) public RegexReplace( Fields fieldDeclaration, String patternString, String replacement ) { super( 1, fieldDeclaration, patternString ); this.replacement = replacement; } @Property(name = "replacement", visibility = Visibility.PUBLIC) @PropertyDescription("The string replacement value.") public String getReplacement() { return replacement; } @Property(name = "replaceAll", visibility = Visibility.PUBLIC) @PropertyDescription("Will replace all occurrences of pattern.") public boolean isReplaceAll() { return replaceAll; } @Override public void prepare( FlowProcess flowProcess, OperationCall<Pair<Matcher, TupleEntry>> operationCall ) { TupleEntry tupleEntry = new TupleEntry( operationCall.getDeclaredFields(), Tuple.size( 1 ) ); operationCall.setContext( new Pair<>( getPattern().matcher( "" ), tupleEntry ) ); } @Override public void operate( FlowProcess flowProcess, FunctionCall<Pair<Matcher, TupleEntry>> functionCall ) { String value = functionCall.getArguments().getString( 0 ); if( value == null ) value = ""; TupleEntry output = functionCall.getContext().getRhs(); Matcher matcher = functionCall.getContext().getLhs().reset( value ); if( replaceAll ) output.setString( 0, matcher.replaceAll( replacement ) ); else output.setString( 0, matcher.replaceFirst( replacement ) ); functionCall.getOutputCollector().add( output ); } @Override public boolean equals( Object object ) { if( this == object ) return true; if( !( object instanceof RegexReplace ) ) return false; if( !super.equals( object ) ) return false; RegexReplace that = (RegexReplace) object; if( replaceAll != that.replaceAll ) return false; if( replacement != null ? !replacement.equals( that.replacement ) : that.replacement != null ) return false; return true; } @Override public int hashCode() { int result = super.hashCode(); result = 31 * result + ( replacement != null ? replacement.hashCode() : 0 ); result = 31 * result + ( replaceAll ? 1 : 0 ); return result; } }