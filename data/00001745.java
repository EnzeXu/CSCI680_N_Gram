public class FailOnMissingSuccessFlowListener implements FlowListener { @Override public void onStarting( Flow flow ) { Map<String, Tap> sources = flow.getSources(); for( Map.Entry<String, Tap> entry : sources.entrySet() ) { String key = entry.getKey(); Tap value = entry.getValue(); Set<Hfs> taps = Util.createIdentitySet(); accumulate( taps, value ); for( Hfs tap : taps ) { if( !testExists( flow, tap ) ) throw new FlowException( "cannot start flow: " + flow.getName() + ", _SUCCESS file missing in tap: '" + key + "', at: " + value.getIdentifier() ); } } } public boolean testExists( Flow flow, Hfs tap ) { try { if( !tap.isDirectory( flow.getFlowProcess() ) ) return true; return new Hfs( new TextLine(), new Path( tap.getPath(), "_SUCCESS" ).toString() ).resourceExists( flow.getFlowProcess() ); } catch( IOException exception ) { throw new FlowException( exception ); } } public void accumulate( Set<Hfs> taps, Tap value ) { if( value == null ) return; if( value instanceof Hfs ) taps.add( (Hfs) value ); else if( value instanceof PartitionTap ) taps.add( (Hfs) ( (PartitionTap) value ).getParent() ); else if( value instanceof MultiSourceTap ) iterate( taps, (MultiSourceTap) value ); else throw new IllegalArgumentException( "unsupprted Tap type: " + value.getClass().getName() ); } public void iterate( Set<Hfs> taps, MultiSourceTap value ) { Iterator<Tap> childTaps = value.getChildTaps(); while( childTaps.hasNext() ) accumulate( taps, childTaps.next() ); } @Override public void onStopping( Flow flow ) { } @Override public void onCompleted( Flow flow ) { } @Override public boolean onThrowable( Flow flow, Throwable throwable ) { return false; } }