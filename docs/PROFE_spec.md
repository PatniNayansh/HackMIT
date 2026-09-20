
--- Page 1 ---
PROFE  —  Project  Spec  
HackMIT  2026  ·  9.19–9.20  
A  presentation  evaluation  tool  that  measures  two  things  no  existing  product  measures:  where  a  
viewer's
 
eye
 
actually
 
goes,
 
and
 
what
 
different
 
viewers
 
actually
 
take
 
away.
 
 
1.  Thesis  
Presentation  feedback  tools  check  delivery  and  formatting  —  filler  words,  pace,  eye  contact,  
font
 
size.
 
None
 
check
 
whether
 
the
 
audience
 
extracts
 
the
 
point
 
the
 
presenter
 
meant,
 
and
 
none
 
check
 
whether
 
that
 
extraction
 
differs
 
depending
 
on
 
who
 
is
 
watching
.
 
PROFE  simulates  three  audiences  separated  by  prior  knowledge ,  has  each  one  attempt  
comprehension
 
of
 
a
 
slide,
 
and
 
measures
 
divergence
 
between
 
them.
 
High
 
novice-to-expert
 
divergence
 
localizes
 
the
 
expert
 
blind
 
spot
 
to
 
a
 
specific
 
slide
 
—
 
the
 
single
 
most
 
common
 
failure
 
mode
 
in
 
technical
 
presentations,
 
and
 
one
 
the
 
presenter
 
is
 
structurally
 
unable
 
to
 
detect
 
alone.
 
A  saliency  layer  adds  where  the  eye  goes .  An  optional  neural  layer  adds  which  cortical  systems  
the
 
slide
 
drives
.
 
One  sentence:  We  show  you  the  wrong  conclusion  your  audience  will  draw,  and  which  slide  
caused
 
it.
 
 
2.  Prior  art  —  what  the  research  turned  up  
2.1  The  generic  version  is  a  crowded  commodity  "Upload  a  talk,  get  feedback  on  engagement  and  clarity"  is  a  mature  category.  Yoodli  tracks  12+  
metrics
 
including
 
pacing,
 
filler
 
words,
 
eye
 
contact
 
and
 
clarity.
 
Orai,
 
Poised,
 
Insight7,
 
Yudly,
 
VirtualSpeech
 
and
 
Presentace
 
occupy
 
the
 
same
 
space.
 
Microsoft
 
ships
 
Speaker
 
Coach
 
free
 
inside
 
PowerPoint.
 
Implication:  do  not  pitch  a  presentation  coach.  Pitch  the  divergence  measurement.  
--- Page 2 ---
2.2  A  published  null  result  sits  directly  on  the  naive  neural  approach  arXiv  2607.01400  (July  2026)  ran  TRIBE  on  48  YouTube  videos,  reduced  the  predicted  cortical  
response
 
to
 
a
 
scalar
 
engagement
 
curve
 
via
 
global
 
field
 
power,
 
and
 
tested
 
it
 
against
 
"most
 
replayed"
 
heatmaps.
 
Signal  Pooled  partial  r  (position-controlled)  
TRIBE  engagement  (GFP)  +0.058  (95%  CI  −0.04  to  0.15,  p  =  0.23)  
Loudness  baseline  +0.040  
Motion  baseline  −0.061  The  null  held  across  six  cortical-network  readouts  (visual,  auditory,  salience,  frontal,  parietal,  
whole-cortex)
 
and
 
under
 
a
 
circular-shift
 
permutation
 
test
 
preserving
 
autocorrelation.
 
Video-level
 
ranking
 
also
 
failed:
 
ρ
 
with
 
views
 
and
 
likes
 
near
 
zero
 
and
 
slightly
 
negative.
 
Implication:  a  scalar  "engagement  score"  derived  from  predicted  brain  drive  is  a  known-false  
output.
 
It
 
must
 
not
 
appear
 
in
 
the
 
product.
 
2.3  But  the  measured-brain  literature  is  solidly  positive  -  Meta-analysis  of  14  studies:  intersubject  correlation  (ISC)  correlates  with  attention  at  r  =  
0.65
.
 -  Cohen  et  al.  (2018):  EEG  ISC  during  educational  videos  predicted  information  retention;  
when
 
attention
 
was
 
divided
 
by
 
a
 
counting
 
task,
 
ISC
 
dropped
 
significantly
 
and
 
stopped
 
predicting
 
exam
 
performance.
 -  Classroom  EEG  studies  reproduce  ISC  engagement  effects  with  portable  
consumer-grade
 
equipment.
 
The  gap  between  2.2  and  2.3  is  a  readout  problem.  GFP  measures  amplitude ;  ISC  measures  
reliability
.
 
Loud,
 
flashy
 
content
 
produces
 
amplitude.
 
Only
 
followable
 
content
 
produces
 
synchrony.
 
2.4  Additional  constraint  —  the  seductive  allure  effect  Brain  images  and  neuroscience  language  increase  perceived  credibility  of  an  explanation  
independent
 
of
 
whether
 
the
 
explanation
 
is
 
sound.
 
Any
 
cortical
 
visualization
 
we
 
ship
 
will
 
do
 
this
 
to
 
judges.
 
Naming
 
this
 
ourselves
 
in
 
the
 
methods
 
panel
 
converts
 
it
 
from
 
a
 
liability
 
into
 
a
 
credibility
 
signal.
 
 
--- Page 3 ---
3.  What  we  build  
Three  layers,  two  speeds  
Layer  What  it  answers  Speed  Demo  path  
Saliency  Where  does  the  eye  land  first?  
<  1  s,  CPU  Live  
Simulated  audiences  
What  do  three  viewers  think  the  point  is?  
<  5  s  Live  
Neural  Which  cortical  systems  does  this  slide  drive?  
6–13  min  GPU/slide  Precomputed  only  
The  live  path  is  the  product.  The  neural  layer  is  the  proof-of-concept  centerpiece,  run  overnight  
on
 
two
 
bundled
 
sample
 
decks.
 
Core  design  rules  —  non-negotiable  1.  Audiences  perform,  they  never  rate.  No  clarity:  7/10 anywhere  in  the  codebase.  
An
 
audience
 
outputs
 
a
 
takeaway,
 
a
 
confidence,
 
and
 
unresolved
 
terms.
 
Those
 
are
 
checkable;
 
a
 
rating
 
is
 
not. 2.  Every  number  decomposes  into  its  source  text.  Clicking  a  score  shows  the  three  
takeaways
 
that
 
produced
 
it.
 3.  No  scalar  engagement  metric.  See  §2.2.  If  one  appears,  it  is  a  bug.  GFP  is  reproduced  
only
 
as
 
a
 
labeled
 
negative
 
baseline
 
in
 
the
 
methods
 
panel.
 4.  Neural  output  is  labeled  "predicted  —  simulated,  not  measured"  on  the  image  itself,  
not
 
in
 
a
 
tab.
 5.  Ratios  within  a  deck,  never  absolutes.  Within-deck  comparison  holds  presenter,  topic  
and
 
recording
 
conditions
 
fixed
 
and
 
cancels
 
much
 
of
 
the
 
amplitude
 
confound.
 
 
4.  The  three  audiences  
Defined  only  by  prior  knowledge .  Not  by  personality,  demographic  or  vibe  —  prior  knowledge  
is
 
the
 
axis
 
with
 
theory
 
behind
 
it.
 
--- Page 4 ---
Persona  Definition  
novice No  domain  background,  general  education  
peer Adjacent  technical  field,  not  this  subfield  
expert Deep  in  this  exact  subfield  Each  receives  the  slide  image,  extracted  text,  and  running  deck  context.  Each  returns  strict  
JSON:
 
{    "takeaway":  "one  sentence  —  what  they  think  the  point  is",    "confidence":  0.0,    "unresolved_terms":  ["terms  used  but  not  defined  for  THIS  audience"],    "questions":  ["what  they'd  need  to  ask  to  follow"],    "inferred_claim":  "what  they  think  the  presenter  wants  them  to  believe"  }  Run  concurrently.  Cache  by  (slide_hash,  persona).  Never  let  a  persona  see  another  
persona's
 
output.
 
Derived  metrics  -  intent_alignment —  semantic  similarity  between  each  audience's  takeaway  and  the  
presenter's
 
declared
 
intent.
 
Three
 
values
 
per
 
slide. -  audience_divergence —  pairwise  semantic  distance  between  the  three  takeaways.  
This
 
is
 
the
 
headline
 
metric.
 -  blind_spot_score —  expert  alignment  minus  novice  alignment.  Large  positive  =  
classic
 
expert
 
blind
 
spot. -  term_gap —  terms  in  novice.unresolved_terms absent  from  expert.unresolved_terms.  The  specific  words  costing  you  the  room. 
 
5.  Neural  layer  (precomputed)  
What  we  extract:  where ,  not  how  much  The  valuable  signal  from  TRIBE  is  spatial.  Collapsing  to  a  scalar  throws  it  away  and  reproduces  
a
 
known
 
null.
 
Question  per  slide:  does  it  drive  language  regions,  or  only  early  visual?  
--- Page 5 ---
A  slide  producing  visual-cortex  drive  with  little  language  drive  is  one  people  are  looking  at  
without
 
processing
 
meaning
 
from
 
—
 
decorative,
 
dense,
 
pretty,
 
empty.
 
A
 
slide
 
driving
 
language
 
and
 
association
 
cortex
 
is
 
being
 
read
 
and
 
understood.
 
processing_ratio =  language_drive  /  visual_drive,  expressed  as  a  Z-score  relative  to  
the
 
deck
 
mean.
 
Slide  12  drives  language  40%  below  deck  average  —  a  claim  with  content.  Slide  12  
scores
 
61
 
—
 
not.
 
This  ratio  is  a  proposed  readout,  not  a  validated  metric ,  and  the  write-up  must  say  so.  
Hard  constraints  -  TRIBE  requires  narrated  slides.  On  a  static  image  V-JEPA2  contributes  nearly  nothing  
—
 
signal
 
comes
 
almost
 
entirely
 
from
 
the
 
audio
 
and
 
transcript
 
encoders.
 
A
 
silent
 
PDF
 
gives
 
TRIBE
 
nothing.
 
Disable
 
the
 
layer
 
entirely
 
and
 
say
 
so
 
rather
 
than
 
fabricating
 
from
 
text
 
alone.
 -  6–13  min  GPU  per  slide.  Live  inference  is  impossible.  Precompute  only,  via  a  
standalone
 
resumable
 
CLI.
 -  Out  of  distribution.  TRIBE  was  trained  on  movie-watching  fMRI.  Narrated  slides  are  not  
movies.
 
State
 
this.
 
Convergence  The  processing  ratio  and  the  novice/expert  divergence  measure  related  things  through  
completely
 
independent
 
routes
.
 
Where
 
they
 
agree
 
→
 
converging
 
evidence.
 
Where
 
they
 
disagree
 
→
 
something
 
interesting
 
to
 
say.
 
 
6.  Validation  —  the  thing  that  separates  us  from  a  demo  
A  /validate route  that  shows  a  slide  and  collects  (a)  a  one-sentence  takeaway  and  (b)  a  
self-reported
 
familiarity
 
rating,
 
appending
 
to
 validation/{deck_id}.json. 
On-site  protocol:  two  slides,  ~8  hackers  who  know  the  topic,  ~8  who  don't,  one  sentence  each.  
Compare
 
real
 
novice/expert
 
divergence
 
against
 
simulated
 
divergence.
 
Twenty  minutes  of  work.  It  is  the  difference  between  "we  made  LLM  personas"  and  "we  made  
LLM
 
personas
 
and
 
checked
 
whether
 
they
 
behave
 
like
 
real
 
people."
 
Recruit  during  the  2am  lull  when  everyone  wants  an  excuse  to  stop  coding.  
 
--- Page 6 ---
7.  Build  order  
Respect  this  order.  Each  step  is  a  working  checkpoint.  
1.  audiences.py +  divergence.py with  tests.  Test  on  two  hand-written  slides  where  
the
 
answer
 
is
 
known:
 
one
 
obviously
 
clear
 
(all
 
three
 
converge),
 
one
 
loaded
 
with
 
undefined
 
jargon
 
(novice
 
must
 
diverge).
 
If
 
those
 
two
 
don't
 
behave
 
correctly,
 
the
 
project
 
is
 
unsound
 
—
 
fail
 
loudly
 
here.
 2.  Slide  detail  screen  wired  to  those  modules.  PDF  ingest  only.  3.  saliency.py +  scanpath.py with  synthetic-image  tests  (one  white  square  on  black  
→
 
one
 
fixation
 
at
 
its
 
center). 4.  diagnose.py rules. 5.  Blind  spot  view.  6.  /validate route. 7.  pptx  ingest.  8.  fix.py —  the  revise-and-rescore  loop. 9.  Neural  layer,  only  if  1–6  are  done.  
Cut  order  under  time  pressure  Cut  from  the  bottom.  If  you  must  cut,  cut  the  fix  loop  and  the  neural  layer  —  never  the  
audiences.
 
Saliency
 
alone
 
is
 
a
 
nice
 
linter;
 
divergence
 
is
 
a
 
finding.
 
Hard  checkpoint  If  step  1's  jargon  slide  does  not  produce  novice/expert  divergence  by  hour  3 ,  the  personas  
aren't
 
differentiating.
 
Stop
 
and
 
fix
 
the
 
prompting,
 
or
 
the
 
rest
 
of
 
the
 
build
 
rests
 
on
 
nothing.
 
 
8.  Technical  stack  
Python  FastAPI  +  React/Vite.  Single  repo,  make  dev runs  both. 
Ingest:  python-pptx for  pptx,  pdfplumber +  pdf2image for  pdf.  Render  slides  to  1280×720  
PNG.
 
Cap
 
30
 
slides. 
Saliency:  SaliencyModel protocol  with  three  implementations  behind  it  — 
-  SpectralResidual (OpenCV)  —  CPU,  instant,  zero  deps.  Default  and  demo  path.  -  DeepGaze —  DeepGaze  IIE  if  weights  present,  else  clear  startup  error. -  Stub —  deterministic,  for  tests. 
--- Page 7 ---
Scanpath:  greedy  max  +  inhibition-of-return  (zero  a  σ=60 px  Gaussian),  N=6  fixations.  
Document
 
that
 
this
 
approximates
 
a
 
scanpath;
 
it
 
is
 
not
 
a
 
validated
 
scanpath
 
model.
 
Semantic  similarity:  sentence-transformers,  local  model,  no  API  call. 
Neural:  NeuralModel protocol  →  (T,  20484) on  fsaverage5,  TR=1s.  CachedNeural is  the  
only
 
path
 
available
 
to
 
the
 
web
 
app;
 TribeNeural raises  immediately  if  called  from  the  FastAPI  
process.
 
Parcellation
 
via
 nilearn.datasets (Schaefer  or  Destrieux).  Rendering  via  nilearn.plotting.plot_surf_stat_map,  colormap  cold_hot,  symmetric  about  zero. 
Offline:  fully  offline  capable  except  LLM  calls.  Bundle  cached  audience  responses  for  sample  
decks
 
so
 
the
 
demo
 
survives
 
dead
 
wifi.
 
 
9.  Screens  
1.  Upload  —  drag-drop,  plus  two  bundled  sample  decks  (one  clean,  one  with  a  deliberate  
expert
 
blind
 
spot),
 
fully
 
cached.
 2.  Deck  overview  —  thumbnail  grid,  two  small  bars  per  slide  (attention  alignment,  
audience
 
divergence).
 
Sort
 
worst
 
divergence
 
first.
 3.  Slide  detail  —  split  view.  Left:  slide  with  saliency  heatmap  (viridis,  adjustable  opacity)  
and
 
numbered
 
fixation
 
path.
 
Right:
 
three
 
audience
 
cards,
 
each
 
showing
 
its
 
takeaway
 
verbatim,
 
confidence,
 
unresolved
 
terms.
 
Declared
 
intent
 
above
 
them
 
for
 
comparison.
 4.  Blind  spot  view  —  deck-level  chart:  novice  vs.  expert  alignment  per  slide,  widest  gaps  
called
 
out.
 
This
 
is
 
the
 
screen
 
that
 
sells
 
the
 
demo.
 5.  Neural  view  —  sample  decks  only.  Cortical  surface,  lateral  +  medial,  both  hemispheres.  
Permanent
 
overlay:
 
"Predicted
 
response
 
—
 
simulated,
 
not
 
measured."
 
Processing
 
ratio
 
as
 
deviation
 
from
 
deck
 
mean,
 
deck
 
distribution
 
shown
 
behind
 
it.
 
Clear
 
empty
 
state
 
when
 
no
 
cached
 
data
 
—
 
never
 
a
 
placeholder
 
brain.
 6.  Fix  view  —  before/after  on  both  metrics.  If  the  fix  doesn't  improve,  show  the  failed  
attempt
 
and
 
say
 
so.
 7.  Methods  —  see  §10.  
Design:  editorial,  high  contrast,  restrained.  Slide  image  is  the  hero.  Three  audience  cards  read  
as
 
three
 
distinct
 
voices
 
—
 
differentiate
 
by
 
typography
 
and
 
a
 
subtle
 
left-border
 
accent,
 
not
 
cartoon
 
avatars.
 
Monospace
 
for
 
all
 
numbers.
 
Viridis
 
or
 
inferno
 
for
 
heatmaps,
 
never
 
rainbow.
 
No
 
gradients,
 
no
 
emoji.
 
 
10.  Methods  panel  —  required  disclosures  
Ship  all  of  these.  They  are  what  make  the  project  defensible  rather  than  impressive-looking.  
--- Page 8 ---
-  Saliency  predicts  bottom-up  attention ,  not  comprehension  or  aesthetics.  -  The  scanpath  is  an  inhibition-of-return  approximation,  not  a  validated  scanpath  model.  -  Simulated  audiences  are  LLM  personas ,  not  validated  against  human  readers  except  
where
 
§6
 
data
 
exists.
 -  TRIBE  was  trained  on  movie-watching  fMRI;  narrated  slides  are  out  of  distribution .  -  Neural  inference  costs  6–13  min  GPU/slide;  all  neural  results  shown  are  precomputed .  -  The  language/visual  ratio  is  a  proposed  readout,  not  a  validated  metric .  -  Brain  images  increase  perceived  credibility  independent  of  the  underlying  claim's  
validity.
 
State
 
what
 
the
 
image
 
does
 
and
 
does
 
not
 
establish.
 -  Reproduce  the  GFP  readout  as  a  labeled  negative  baseline  alongside  the  published  null  
it
 
replicates.
 
 
11.  Sponsor  targeting  
Sponsor  Fit  Notes  
Long  Lake  Strongest  A  skeptic  shown  a  wrong  conclusion  drawn  from  their  own  slide  is  exactly  the  "one  great  experience  away"  brief.  Lead  here.  
Voloridge  Strong  Signal-vs-noise  on  public  data;  we  separate  a  real  signal  from  a  position  artifact.  $5,000  first  place.  
ASUS  Clean  If  the  GX10  is  used  for  the  precompute  run.  Claim  it  in  hour  one  —  first  come,  first  serve.  
Dropbox  Good  "Turn  files  into  action"  —  a  deck  becomes  a  diagnosis.  Fits  the  brief  better  than  most.  
Arrowstreet  Possible  Judged  on  evidence  quality,  source  citation  and  conclusion  confidence.  Our  CIs  and  baselines  map  well,  
--- Page 9 ---
Sponsor  Fit  Notes  
but  the  greenwashing  topic  is  fixed.  Only  with  spare  submission  capacity.  
Token  Company  Cheap  Document  caching  and  model-tiering  on  the  persona  calls.  Three  personas  ×  30  slides  is  a  real  cost  story.  $500.  
Deepgram  Cheap  Transcription  for  narrated  decks.  
Meta  Weak  Brief  is  human  connection,  not  education  tooling.  Worth  noting  TRIBE  is  Meta  research  (d'Ascoli,  King  et  al.),  but  that  doesn't  fix  the  mismatch.  
ElevenLabs  Drop  Their  criteria  explicitly  deprioritize  simple  TTS.  Only  viable  with  an  agentic  re-narration  loop.   
12.  Demo-night  checklist  
 Two  sample  decks  load  in  <  1  s  with  fully  cached  audience  responses   Works  with  wifi  disabled   Neural  precompute  finished,  or  the  neural  view  cleanly  hidden   Blind  spot  view  is  the  first  screen  a  judge  sees  after  upload   Validation  overlay  populated  with  real  human  data   "Predicted  —  simulated,  not  measured"  visible  on  every  cortical  image   2-minute  pitch  rehearsed  to  muscle  memory   Demo  video  recorded  as  backup  before  the  venue  gets  loud  
 
--- Page 10 ---
13.  The  pitch  
Presentation  tools  check  your  delivery.  Nobody  can  tell  you  whether  your  content  
landed
 
—
 
or
 
that
 
it
 
landed
 
differently
 
for
 
the
 
person
 
who
 
already
 
knew
 
the
 
topic
 
than
 
for
 
the
 
one
 
who
 
didn't.
 
PROFE  simulates  three  audiences  separated  by  prior  knowledge,  has  each  
one
 
tell
 
us
 
what
 
they
 
think
 
your
 
point
 
was,
 
and
 
shows
 
you
 
where
 
they
 
disagree.
 
That
 
disagreement
 
is
 
your
 
expert
 
blind
 
spot,
 
localized
 
to
 
a
 
single
 
slide.
 
We  tested  a  brain-encoding  model  as  a  fourth  signal,  found  the  published  scalar  
readout
 
fails,
 
and
 
report
 
a
 
spatial
 
one
 
instead
 
—
 
labeled
 
honestly
 
as
 
the
 
hypothesis
 
it
 
is.
 
What  to  claim:  we  found  the  wrong  conclusion  your  audience  draws.  What  not  to  claim:  a  
number
 
that
 
says
 
how
 
engaging
 
your
 
slide
 
is.
 
 