# IMPORTANT  Work on release 9 is in progress.  DO NOT checkout from the repository, get release 8 from 
https://github.com/MiloMindbender/UE4VirtualProduction/releases)

# VPStudio, my Unreal Virtual Production tutorial project

VPStudio is a tutorial (for you) and learning (for me!) project.  Everything in my older projects is in here. Feel free to use anything in your own projects without license or restrictions. Please credit me, Greg Corson, for helping you if you can.  

For updates, tutorials and demos please [subscribe to my youtube channel](https://www.youtube.com/user/GregCorson). You can also ask for help on [this discord channel](https://discord.gg/ReEhkhc)or this [facebook group](https://www.facebook.com/groups/virtualproduction)

# Compatibility

This release REQUIRES Unreal 4.26 or higher.

It appears to work on 4.27 preview but WILL NOT be fully tested till the release version.

In 5.0.0 early access 2 video input stops working when you press play, so you can't use it.

# What's new in Release 9 [latest release here](https://github.com/MiloMindbender/UE4VirtualProduction/releases)

* Added roll adjustment to autorig.

# What's new in Release 8 [latest release here](https://github.com/MiloMindbender/UE4VirtualProduction/releases)

* ActorTransformTelemetry replaces ALL the other telemetry sending blueprints.  See [this document](https://github.com/MiloMindbender/UE4VirtualProduction/blob/master/VPStudio/TelemetryViewer/README.md) for datails on how to use it.
* ActorTransformTelemetry runs "post physics tick" so all actors are updated before it sends their positions.
* LiveLinkTracker supports trackers that send relative data.  Select tracker type "relative tracker" and supply "Relative Tracker Offset" to align tracking to the coordinate system of the Talent Mark.
* Autorig no longer displays a vive puck at the top.  This is unnessary because the LiveLinkTracker draws the tracker mesh.
* Autorig sizes it's rods to the exact rig measurements you enter so they don't stick out.
* Autorig's rods are now colored to match unreal conventions for x, y, z
* Added ThinAxis actor, makes it easier to see the orign of smaller objects, just attach it.
* Vive controller buttons work when using livelink.  They still need to be mapped in the Vive control panel and the Unreal 3d window has to have focus.
* Added RecordMeasurements a way to measure your studio using the VIVE controllers/trackers
* Added MeasurementMarker actor, displays an axis marker and current position, used by RecordMeasurements
* The default map is now SINGLE CAMERA.
* Flying logo is now self contained in one actor.  Epic fixed a bug that prevented this in earlier versions
* Flying logo can be started by pressing 9
* Added categories for some actor variables to organize them, also added some tooltips for them and set limits on variables that needed them.

# Bug Fixes


# Changes you won't notice unless you have customized my blueprints

These are mostly changes to make the whole setup work better, simplify the internals and make everything more consistant.

* Renamed media player assets that go with the Flying Screen actor
* Renamed all the parts of the AJA media bundles in a consistant way.
* Renamed Autorig "entrance pupil" transform to "output" for consistancy with other rigs.
* MotionControllerTracker, DelayedOutput has been renamed to "Output" for consistancy with LiveLinkTracker
* All tracker actors use a more efficient way of setting the correct mesh for the tracker.

# TODO for 9

* An Unreal bug with socket snapping that prevented using it to attach things to camera rigs and trackers has supposedly been fixed.  Need to test this as it would simplify the blueprints used for trackers and rigs.
* Need an autorig that supports ballheads better, the current one doesn't have a pivot point below tracker where a ballhead's ball would be.
* How to rotate/scale the live camera or do simple billboard stuff.
* Fix problem with editor window displaying non-square pixels when play is pressed.
* Need workflow for still pictures
* Test syncing of external video image sequence
* Virtual clapper for scene sync, where are assets for take recorder?
* Make touchOSC layout that works with VPStudio default
* Neuron mocap recording tutorial
* Test telemetry with new telemetry viewer

# Important changes starting with Release 8

Since release 8 I've made a few changes to the default setup and documentation.  The default is now ONE camera.  If you need TWO [read this](https://github.com/MiloMindbender/UE4VirtualProduction/blob/master/VPStudio/TwoCamera.md) to see how to get the old setup back.

The default setup is VIVE trackers coming in over LiveLink.  If you need to get tracking from "MotionController" components or custom tracking plugins instead they are covered in a section of [this document.](https://github.com/MiloMindbender/UE4VirtualProduction/blob/master/VPStudio/Tracking.md)

The tracker debugger and telemetry viewer has it's own document, [read it here](https://github.com/MiloMindbender/UE4VirtualProduction/blob/master/VPStudio/TelemetryViewer/README.md)

The new tool for measuring your studio has it's own document, [read it here](https://github.com/MiloMindbender/UE4VirtualProduction/blob/master/VPStudio/RecordMeasurements.md) 

# Updating to New Releases BACKUP!

This is a learning and teaching project, not a commercial product so a new release may not do everything the same way as previous the last one. ALWAYS make a copy of your project before updating it to a new VPStudio so if something doesn't work like you expect you can fall back to that working copy.  My release numbers are always whole numbers, the largest one will be the latest.  

The main branch on github is ALWAYS A WORK IN PROGRESS and may have unfinished or broken features!  Almost everyone should get the [latest release from the releases section](https://github.com/MiloMindbender/UE4VirtualProduction/releases) main branch of github is updated FREQUENTLY as I work on it. Using the green button to clone or download a ZIP of the repository from the main github page may get you unfinished and untested code.  Please use the [latest release from the releases section](https://github.com/MiloMindbender/UE4VirtualProduction/releases) or clone the repository from the latest numbered release tag.

VPStudio is NOT A PRODUCT, it is an example that has to be customized for your hardware and studio.  Keep track of the changes you had to make to previous versions.  Usually these are small and can be quickly copied over to the new VPStudio.

I recommend you start with a clean copy of VPStudio, get it running on your hardware and save it.  Don't add or change anything but what you need to do to get it running.  To use it with your own content, make a copy of your working VPStudio and migrate your own content to it. 

# The sample level IS boaring!

A lot of "free" content, is licensed so you can use it in your own games and videos but you CAN NOT redistribute it to other people.  So I can't give you the content used in my demo videos.  VPStudio uses only content built into Unreal.  See the [use your own sets tutorial](https://youtu.be/trlpmm5gI6U) on my YouTube channel for help on using your own content.  Most demos on my channel were done with Epic free content that you can download yourself from their marketplace and use with VPStudio.

If you are an artist and would like to help by contributing a better sample level under creative commons license, please let me know.

# Setup VPStudio

Right now VPStudio is setup for an AJA Kona HDMI video capture card and VIVE trackers.  If you have a BlackMagic DeckLink or other card/webcam you will need to replace the assets found in the Aja folder with ones for your hardware.  Look at [Unreal Pro Video](https://docs.unrealengine.com/en-US/Engine/ProVideoIO/index.html) for how to set up different BlackMagic and Aja cards. [Using WebCams](https://docs.unrealengine.com/en-US/Engine/MediaFramework/HowTo/UsingWebCams/index.html) shows how to use most USB webcams.  The actors in VPStudioCore->Trackers show how to use both LiveLink and MotionControler trackers, if your trackers use a custom plugin you will have to modify one of these. 

Follow [tutorial 1](https://youtu.be/wWPZjX29asM) and [tutorial 2](https://youtu.be/kRUbUaESw80) to make a rig in Unreal that shows how your camera and tracker are mounted.  There is also an "AutoRig" to make this easier to setup.  I recommend VIVE users use LiveLink support as it avoids a lot of complicated STEAM setup.  If you need to setup buttons on controllers the older [Tutorial 3](https://youtu.be/4LjvekNocD4) shows how to setup VIVE controller buttons to control things.   [Tutorial 4](https://youtu.be/UGHjwZ6J13E) shows how to setup the studio and fine tune your camera and tracker alignment 
 
# Using your own levels

[Tutorial 5](https://youtu.be/trlpmm5gI6U) shows how to add your own sets and assets to a copy of VPStudio.
 
You make a copy of a working VPStudio project and then "migrate" your level into it.  Once you have your level migrated, you use the "levels" window in the editor to add VPStudioCore as a sublevel and you are ready to go, see the tutorial for more details.

Some levels may do things with lighting that make them appear too bright or too dark.  For now you're on your own about fixing this, I don't have any good advice (yet).

# Known Problems

Under Edit->Project Settings->Project->Maps & Modes I provide a VPPlayerController that manages all user input and controls everything.  This is required or nothing will work.  If you want to use this project with a level that requires it's own PlayerController you will need to figure out how to combine mine and yours. I also provide a VPGameState and VPGameMode which are currently EMPTY and not required. 

If you change and recompile the VPCamera blueprint, Unreal may remove the cameras from the composure passes.  If your CG cameras stop working after changing something, check each of the VPStudioBackground, GarbageMatte and VPStudioForeground actors and make sure the "camera source" is override and the Target Camera Actor is VPCamera 1 & 2.

# Features

* Commands go through Edit->Project Settings->Engine->Input so they can be changed. You can change the keyboard keys assigned to functions and the speed of movement on this page.  A function can be mapped to almost any input device you have including PC game controllers, joysticks and VR controllers.

* Multiple cameras and video sources are supported.  This lets you have more than one camera view of your talent and live switch between them.  The project is setup for 2 cameras, adding more requires some blueprint and composure changes but I am working to make this simpler.

* When using LiveLink tracking systems, all the cameras and mattes are tracked live in the editor so you can always see the state of your studio.

* You can place several "Talent Markers" where you want your talent to appear in the set.  Pressing a key will teleport your talent between these marks.

* Tracking is all inside "Tracker" actors to make it easier to use different kinds of tracking devices.  This allows easy customization for VIVE, OpenXR, LiveLink and other tracking solutions.  Only the Tracker actor needs to change.

* You can move the talent markers around using keyboard key to get the right placement of them in your level.  This can be done live while the composite is running and you will see the talent moving through the level.

* There are a number of measuring and setup guides to help you measure your real world studio and align it with your virtual set.  This includes a plumbline that always points straight down, "axis guide world" which is always lined up with the world and some tools to automatically measure your camera rig.  TalentMarkMan is 193cm tall or about 6'3", you can rescale it to match the size of your talent.

* The "rig" your camera and trackers are mounted to can be modeled in the system to exactly match your real-world rig.  This includes things like multiple joints that could be moved by a tracker or might need to be "tweaked" if the rig is slightly out of alignment.  A few samples of different rigs are provided.

* There is a garbage matte setup for use with a greenscreen that covers a wall and floor.  If your greenscreen has a different shape, you can model it and add your own mesh for a better fit.  There is a tutorial on my youtube channel that shows how to adjust the garbage matte to match the size of your greenscreen.

* A HUD with guides useful for aligning cameras if your camera doesn't have guides or has no screen.

* You can turn tracking of stationary objects on and off to avoid seeing the jitter that some trackers have when sitting still.


# Keyboard Keys

For the keyboard/mouse/vive tracker commands to work you need to have clicked in the player viewport or PIE window.  To get control back just hit shift-F1 or ESC to quit.

These are what the keyboard keys currently do, if you want to change them go to Edit->Project Settings->Engine->Input.

Right now switching between "Virtual Production Filming" mode and  inspecting the set has to be done by going to the "VPStudio Comp" actor and setting the output pass to "Player Viewport" for filming or "none" for inspecting.  

When you first press PLAY you will either be looking at the composure composite or in inspection mode.  All the movement keys will be setup to move the inspection camera so you don't accidentally move anything else while you are filming.

Depending on what mode you are in, you can move talent markers or the inspection camera around.  Use the standard Unreal W, S, D, A keys for forward, back right and left movemtnt.  E and C moves up and down.  Use the mouse to look around (pan tilt).

To change to Virtual Production camera one, press 1 or left-click the right hand vive controller trackpad.  For camera two, use 2 or click right on the trackpad.

The N key or Vive controller trigger teleports you to the next TalentMark.  You can have as many of these as you want, just drag them into the level and number them 1, 2, 3 and so on.  When you press this key all your tracked objects & cameras will jump to the next mark and your talent will appear in that spot.

The M key switches between VPCamera "raw" views.  This gives you a view from each of your cameras but without any live composite, you will only see the CG part.  (only works when VPStudioComp output is set to NONE)

The I key switches to the inspection camera.  This is just a free floating camera that lets you look around the level while all the cameras and tracked objects are running so you can see if they are working and in the right places. (only works when VPStudioComp output is set to NONE)

The T key toggles between moving the inspection camera and moving the talent marks.

y, u, p, o and k keys will toggle visibility of TalentMark, MeasuringGuide, CameraModel, CameraRig and Tracker actors

When VPStudioComp output is set to "player viewport" you will see the full composite output of live cameras and the CG background.  The 1 and 2 keys switch between VPCamera1 and VPCamera2

Press b or click the right VIVE controller trackpad down to take a screenshot.  This is the same as typing "shot" into the editor command window.

The H key brings up the hud with camera alignment guides.

The L key locks down all the trackers you have set to be "lockable"

# Bugs and Suggestions

I welcome suggestions on how to make this simpler or more efficient.  You can use the github issues tracker or post to my YouTube channel if you find bugs or have feature requests.
